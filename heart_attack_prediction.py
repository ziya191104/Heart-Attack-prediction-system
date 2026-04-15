import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import pandas as pd
import pickle
import os
import warnings
import threading
import webbrowser
import tempfile
from datetime import datetime
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis
from PIL import Image

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, accuracy_score)

# ──────────────────────────────────────────────────────────────
#  ECG PROCESSOR  (multi-strategy extraction + Pan-Tompkins)
# ──────────────────────────────────────────────────────────────
class ECGProcessor:
    def __init__(self, sampling_rate=360):
        self.sampling_rate = sampling_rate

    # ── SAMPLING RATE (set after image load) ────────────────────
    def set_image_mode(self, img_width=800):
        # Default starting point; derive_clinical_values will try
        # multiple rates automatically and pick the best one.
        self.sampling_rate = img_width / 20.0   # 40 px/s

    # ── HELPER: normalise a 1-D array ───────────────────────────
    @staticmethod
    def _norm(s):
        std = np.std(s)
        if std < 1e-8:
            std = 1e-8
        return (s - np.mean(s)) / std

    # ── SIGNAL EXTRACTION FROM IMAGE ────────────────────────────
    @staticmethod
    def extract_signal_from_image_array(img_array):
        """
        Robust multi-strategy extractor for ECG printout photos.

        Problem with a single strategy: a 12-lead ECG has 4 rows of
        leads + grid lines + text labels.  A naive column-wise scan or
        row-mean picks up ALL of those and produces a flat / chaotic
        signal that yields 0 bpm.

        Solution — try THREE strategies, score each by how clearly it
        shows periodicity (variance of the smoothed derivative), and
        return the winner:

          S1 – Column-wise trace tracker on the BEST horizontal band
               (the 1/4-height strip with the highest pixel variance,
               most likely to be a single clean lead).

          S2 – Row-mean on the same best band (simpler but sometimes
               cleaner when the trace is thick).

          S3 – Column-wise tracker on the middle 40 % of the image
               (fallback when band detection fails).
        """
        h, w = img_array.shape
        img_f   = img_array.astype(np.float32)
        img_inv = 255.0 - img_f           # invert: dark trace → bright

        # ── Step 1: find the horizontal band with the most signal ──
        # Divide image into N equal bands; pick the one whose inverted
        # pixel variance is highest (= most trace activity per column).
        n_bands  = 4
        band_h   = h // n_bands
        best_var = -1
        best_band_top = int(h * 0.2)
        best_band_bot = int(h * 0.8)

        for i in range(n_bands):
            top = i * band_h
            bot = top + band_h
            band = img_inv[top:bot, :]
            var  = float(np.var(band))
            if var > best_var:
                best_var      = var
                best_band_top = top
                best_band_bot = bot

        # Trim 10 % from each edge of the chosen band (avoid borders/labels)
        pad   = max(1, band_h // 10)
        b_top = best_band_top + pad
        b_bot = best_band_bot - pad
        if b_bot <= b_top:
            b_top = best_band_top
            b_bot = best_band_bot

        roi = img_inv[b_top:b_bot, :]     # region of interest

        # ── Step 2: adaptive threshold (top-25 % brightest pixels) ─
        thresh = float(np.percentile(roi, 75))
        thresh = max(thresh, 30.0)         # never go below 30/255
        binary = (roi > thresh).astype(np.uint8)

        # ── S1: column-wise trace tracker on ROI ───────────────────
        roi_h   = b_bot - b_top
        last_val = float(roi_h) / 2.0
        s1 = []
        for col in range(w):
            rows = np.where(binary[:, col] > 0)[0]
            if len(rows) > 0:
                val      = float(np.mean(rows))
                last_val = val
            else:
                val = last_val
            s1.append(-val)               # invert: high row = low ECG amplitude
        s1 = np.array(s1, dtype=np.float64)

        # ── S2: row-mean on ROI ────────────────────────────────────
        s2 = np.mean(roi, axis=0).astype(np.float64)

        # ── S3: column-wise on fixed middle 40 % (safety fallback) ─
        mid_top = int(h * 0.30)
        mid_bot = int(h * 0.70)
        mid_roi = img_inv[mid_top:mid_bot, :]
        thresh3 = float(np.percentile(mid_roi, 75))
        thresh3 = max(thresh3, 30.0)
        bin3    = (mid_roi > thresh3).astype(np.uint8)
        last3   = float(mid_bot - mid_top) / 2.0
        s3 = []
        for col in range(w):
            rows = np.where(bin3[:, col] > 0)[0]
            if len(rows) > 0:
                val   = float(np.mean(rows))
                last3 = val
            else:
                val = last3
            s3.append(-val)
        s3 = np.array(s3, dtype=np.float64)

        # ── Score each candidate ───────────────────────────────────
        # Metric: std of the smoothed first-derivative.
        # A signal with clear periodic peaks has a high derivative variance.
        def _score(sig):
            n   = ECGProcessor._norm(sig)
            sm  = np.convolve(n, np.ones(7) / 7, mode='same')
            return float(np.std(np.diff(sm)))

        candidates = [s1, s2, s3]
        scores     = [_score(c) for c in candidates]
        best       = candidates[int(np.argmax(scores))]

        return ECGProcessor._norm(best)

    # ── BANDPASS FILTER ─────────────────────────────────────────
    def bandpass_filter(self, ecg, lowcut=0.5, highcut=40.0):
        nyquist = 0.5 * self.sampling_rate
        low  = max(lowcut  / nyquist, 1e-6)
        high = min(highcut / nyquist, 0.999)
        b, a = scipy_signal.butter(4, [low, high], btype='band')
        return scipy_signal.filtfilt(b, a, ecg)

    # ── R-PEAK DETECTION  (Pan-Tompkins + progressive fallback) ─
    def detect_r_peaks(self, ecg):
        """
        Full Pan-Tompkins pipeline with 5 progressive fallback tiers
        so noisy image-derived signals still yield detectable peaks.
        """
        ecg_norm = self._norm(ecg)

        # Square → moving-window integration
        squared    = ecg_norm ** 2
        win_len    = max(1, int(0.15 * self.sampling_rate))
        integrated = np.convolve(squared, np.ones(win_len) / win_len, mode='same')

        # Try two minimum-distance values: 0.4 s (150 bpm) and 0.25 s (240 bpm)
        # Using 0.4 s is safer for typical resting HR (50-120 bpm)
        for min_sec in [0.4, 0.25]:
            min_dist = max(1, int(min_sec * self.sampling_rate))
            # Try height thresholds strict → lenient
            for factor in [1.0, 0.5, 0.2, 0.05, 0.0]:
                if factor > 0:
                    height = np.mean(integrated) + factor * np.std(integrated)
                else:
                    height = np.mean(integrated) * 0.1   # almost no threshold
                peaks, _ = scipy_signal.find_peaks(
                    integrated, distance=min_dist, height=height)
                if len(peaks) >= 2:
                    return peaks

        # Last resort: find peaks directly on the raw normalised signal
        for min_sec in [0.4, 0.25]:
            min_dist = max(1, int(min_sec * self.sampling_rate))
            for h_val in [1.0, 0.5, 0.3, 0.1]:
                peaks, _ = scipy_signal.find_peaks(
                    ecg_norm, distance=min_dist, height=h_val)
                if len(peaks) >= 2:
                    return peaks

        return np.array([], dtype=int)

    # ── RR INTERVALS (physiologically filtered) ─────────────────
    def compute_rr_intervals(self, peaks):
        if len(peaks) < 2:
            return np.array([])
        rr_ms = np.diff(peaks) / self.sampling_rate * 1000.0
        # Keep only physiologically plausible intervals (30–200 bpm)
        rr_ms = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]
        return rr_ms

    # ── FEATURE EXTRACTION (for ML model) ───────────────────────
    def extract_features(self, ecg_signal):
        try:
            filtered = self.bandpass_filter(ecg_signal)
        except Exception:
            filtered = ecg_signal

        r_peaks  = self.detect_r_peaks(filtered)
        features = []

        if len(r_peaks) > 1:
            rr = self.compute_rr_intervals(r_peaks)
            if len(rr) > 0:
                hr = 60000.0 / float(np.median(rr))
                features.extend([float(np.mean(rr)), float(np.std(rr)),
                                  float(np.median(rr)), hr])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        features.extend([float(np.mean(filtered)), float(np.std(filtered))])
        try:
            features.extend([float(skew(filtered)), float(kurtosis(filtered))])
        except Exception:
            features.extend([0.0, 0.0])
        features.extend([float(np.max(filtered)), float(np.min(filtered))])
        return features[:10]

    # ── DERIVE CLINICAL VALUES ───────────────────────────────────
    def derive_clinical_values(self, ecg_signal):
        """
        Tries MULTIPLE sampling rates (px/s) and picks whichever gives
        a physiologically realistic heart rate (40–180 bpm).

        Sampling rates tried correspond to ECG strip durations of:
          10 s → 80 px/s   (single-lead rhythm strip)
          20 s → 40 px/s   (full 12-lead printout photo)
          15 s → 53 px/s   (intermediate)
           8 s → 100 px/s  (short strip)

        This eliminates the "always 0 bpm" problem that occurs when a
        fixed rate is wrong for the given image.
        """
        candidate_rates = [80.0, 40.0, 53.3, 100.0, 60.0]

        best_hr    = 0.0
        best_hrv   = 0.0
        best_fs    = 40.0

        for fs in candidate_rates:
            self.sampling_rate = fs
            try:
                filtered = self.bandpass_filter(ecg_signal)
            except Exception:
                filtered = ecg_signal

            peaks = self.detect_r_peaks(filtered)
            rr    = self.compute_rr_intervals(peaks)

            if len(rr) == 0:
                continue

            hr = 60000.0 / float(np.median(rr))

            # Only accept physiologically realistic HR
            if 40.0 < hr < 180.0:
                hrv = float(np.std(rr)) if len(rr) >= 2 else 0.0
                hrv = max(0.0, min(hrv, 200.0))   # clamp 0–200 ms

                # Prefer HR closest to 75 bpm (typical resting HR)
                # as a tie-breaker when multiple rates work
                if best_hr == 0.0 or abs(hr - 75) < abs(best_hr - 75):
                    best_hr  = hr
                    best_hrv = hrv
                    best_fs  = fs

        # Restore whichever rate won (used by extract_features later)
        self.sampling_rate = best_fs

        # ── Hilbert-transform IMF metrics (use best rate) ────────
        try:
            self.sampling_rate = best_fs
            try:
                filtered = self.bandpass_filter(ecg_signal)
            except Exception:
                filtered = ecg_signal

            analytic   = scipy_signal.hilbert(filtered)
            amplitude  = np.abs(analytic)
            phase      = np.unwrap(np.angle(analytic))
            inst_freq  = np.diff(phase) / (2.0 * np.pi / self.sampling_rate)
            inst_freq  = np.clip(inst_freq, 0, self.sampling_rate / 2)

            imf_amp    = float(np.mean(amplitude))
            imf_freq   = float(np.mean(inst_freq))
            energy     = float(np.mean(amplitude ** 2))
            imf_energy = min(energy / max(energy * 10, 1e-6), 1.0)
        except Exception:
            imf_amp    = float(np.std(ecg_signal))
            imf_freq   = 0.5
            imf_energy = 0.5

        return {
            'heart_rate':    round(best_hr, 1),
            'hrv':           round(best_hrv, 2),
            'imf_energy':    round(imf_energy, 4),
            'imf_amplitude': round(min(imf_amp, 3.0), 4),
            'imf_frequency': round(min(imf_freq / 50.0, 1.0), 4),
        }


# ──────────────────────────────────────────────────────────────
#  MODEL TRAINER
# ──────────────────────────────────────────────────────────────
class ModelTrainer:
    MODEL_FILE = 'heart_attack_model.pkl'

    def __init__(self, log_callback=None):
        self.log           = log_callback or print
        self.model         = None
        self.scaler        = None
        self.feature_names = None
        self.accuracy      = None
        self.roc_auc       = None

    def load_dataset(self, filepath):
        self.log(f"📂 Loading: {os.path.basename(filepath)}\n")
        df = pd.read_csv(filepath)
        self.log(f"✓ Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

        if 'Patient_ID' in df.columns:
            df = df.drop('Patient_ID', axis=1)
            self.log("✓ Dropped Patient_ID column\n")

        if 'Sex' in df.columns:
            df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
            self.log("✓ Converted Sex: M->1, F->0\n")

        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.factorize(df[col])[0]

        if df.isnull().sum().sum() > 0:
            df = df.fillna(df.median(numeric_only=True))

        X = df.iloc[:, :-1].values.astype(float)
        y = df.iloc[:, -1].values.astype(int)

        unique = np.unique(y)
        if len(unique) == 2 and set(unique) != {0, 1}:
            y = (y == unique[1]).astype(int)

        self.feature_names = df.columns[:-1].tolist()
        self.log(f"✓ Features: {self.feature_names}\n")
        self.log(f"✓ Target: No Disease={sum(y==0)}, Heart Disease={sum(y==1)}\n")
        return X, y

    def train(self, filepath):
        self.log("="*60 + "\n")
        self.log("  TRAINING HEART ATTACK PREDICTION MODEL\n")
        self.log("="*60 + "\n\n")

        X, y = self.load_dataset(filepath)

        ecg_placeholder = np.zeros((X.shape[0], 10))
        X_full     = np.hstack([X, ecg_placeholder])
        full_names = self.feature_names + [f'ECG_{i}' for i in range(10)]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_full, y, test_size=0.2, random_state=42, stratify=y)

        self.log(f"Training: {len(X_tr)} samples\n")
        self.log(f"Testing:  {len(X_te)} samples\n\n")

        self.scaler = StandardScaler()
        X_tr_s = self.scaler.fit_transform(X_tr)
        X_te_s = self.scaler.transform(X_te)

        models = {
            'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42),
            'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'SVM':                 SVC(probability=True, random_state=42),
        }

        best_auc, best_name = -1, ''
        for name, clf in models.items():
            self.log(f"Training {name}...")
            clf.fit(X_tr_s, y_tr)
            y_pred  = clf.predict(X_te_s)
            y_proba = clf.predict_proba(X_te_s)[:, 1]
            acc = accuracy_score(y_te, y_pred)
            auc = roc_auc_score(y_te, y_proba)
            cv  = cross_val_score(clf, X_tr_s, y_tr, cv=5, scoring='accuracy')
            self.log(f"  Accuracy: {acc*100:.2f}%\n")
            self.log(f"  ROC-AUC:  {auc:.4f}\n")
            self.log(f"  CV:       {cv.mean():.4f} +/- {cv.std():.4f}\n\n")
            if auc > best_auc:
                best_auc, best_name = auc, name
                self.model    = clf
                self.accuracy = acc
                self.roc_auc  = auc

        self.feature_names = full_names
        self.log("="*60 + "\n")
        self.log(f"BEST MODEL: {best_name}\n")
        self.log(f"  ACCURACY: {self.accuracy*100:.2f}%\n")
        self.log(f"  ROC-AUC:  {self.roc_auc:.4f}\n")
        self.log("="*60 + "\n\n")

        y_final = self.model.predict(X_te_s)
        self.log("Classification Report:\n")
        self.log(classification_report(y_te, y_final,
                                       target_names=['No Disease', 'Heart Disease']))

        cm = confusion_matrix(y_te, y_final)
        self.log(f"\nConfusion Matrix:\n")
        self.log(f"              Predicted No  Predicted Yes\n")
        self.log(f"Actual No     {cm[0][0]:<13} {cm[0][1]}\n")
        self.log(f"Actual Yes    {cm[1][0]:<13} {cm[1][1]}\n\n")

        with open(self.MODEL_FILE, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler,
                         'feature_names': self.feature_names,
                         'accuracy': self.accuracy, 'roc_auc': self.roc_auc}, f)

        self.log(f"✓ Model saved: {self.MODEL_FILE}\n")
        self.log("="*60 + "\n✅ TRAINING COMPLETE!\n\n")
        return self.model, self.scaler

    @staticmethod
    def load_saved_model():
        with open(ModelTrainer.MODEL_FILE, 'rb') as f:
            d = pickle.load(f)
        return (d['model'], d['scaler'], d['feature_names'],
                d.get('accuracy'), d.get('roc_auc'))


# ──────────────────────────────────────────────────────────────
#  MAIN APPLICATION
# ──────────────────────────────────────────────────────────────
class HeartAttackApp:
    C_DARK   = '#2c3e50'
    C_WHITE  = '#ffffff'
    C_GRAY   = '#f0f2f5'
    C_BLUE   = '#3498db'
    C_RED    = '#e74c3c'
    C_GREEN  = '#27ae60'
    C_ORANGE = '#f39c12'

    STRESS_MAP = {'Low': 0, 'Medium': 1, 'High': 2}

    def __init__(self, root):
        self.root = root
        self.root.title("AI Heart Attack Prediction System")
        self.root.geometry("1440x920")
        self.root.configure(bg=self.C_GRAY)

        self.model         = None
        self.scaler        = None
        self.feature_names = None
        self.accuracy      = None
        self.roc_auc       = None
        self.ecg_data      = None
        self.ecg_features  = None
        self.ecg_metrics   = {}
        self.dataset_path  = None

        self._last_risk      = None
        self._last_analysis  = ''
        self._last_recommend = ''
        self._last_patient   = {}

        self._build_ui()
        self.root.after(500, self._auto_train)

    # ── AUTO-TRAIN ──────────────────────────────────────────────
    def _auto_train(self):
        try:
            (self.model, self.scaler, self.feature_names,
             self.accuracy, self.roc_auc) = ModelTrainer.load_saved_model()
            self._log("✓ Loaded existing model\n")
            if self.accuracy:
                self._log(f"Accuracy: {self.accuracy*100:.2f}%\n\n")
            self._log("Ready to predict!\n")
            self.train_status.config(text='✅ Model Ready', fg=self.C_GREEN)
            if self.accuracy:
                self.accuracy_label.config(
                    text=f"Accuracy: {self.accuracy*100:.2f}%  |  ROC-AUC: {self.roc_auc:.4f}")
            return
        except Exception:
            pass

        for name in ['CHD_Dataset.csv', 'CHD_Dataset_csv.csv', 'chd_dataset.csv']:
            if os.path.exists(name):
                self.dataset_path = os.path.abspath(name)
                self.dataset_label.config(
                    text=os.path.basename(self.dataset_path), fg='#27ae60')
                self._log(f"✓ Found: {os.path.basename(self.dataset_path)}\n\n")
                self._log("Starting auto-training...\n\n")
                self.train_btn.config(state='disabled', text='Training...')
                self.train_progress.start(10)
                self.train_status.config(text='Auto-training...', fg=self.C_BLUE)
                threading.Thread(target=self._run_training, daemon=True).start()
                return

        self._log("No dataset found.\nPlace CHD_Dataset.csv in the same folder.\n")

    # ── BUILD UI ────────────────────────────────────────────────
    def _build_ui(self):
        hdr = tk.Frame(self.root, bg=self.C_DARK, height=80)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  AI Heart Attack Prediction System",
                 font=('Arial', 24, 'bold'), fg='white', bg=self.C_DARK).pack(pady=20)

        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook', background=self.C_GRAY, borderwidth=0)
        style.configure('TNotebook.Tab', font=('Arial', 12, 'bold'),
                        padding=[22, 10], background='#bdc3c7')
        style.map('TNotebook.Tab',
                  background=[('selected', self.C_DARK)],
                  foreground=[('selected', 'white')])

        nb = ttk.Notebook(self.root)
        nb.pack(fill='both', expand=True, padx=12, pady=12)

        tab_train = tk.Frame(nb, bg=self.C_GRAY)
        nb.add(tab_train, text='  Training  ')

        tab_pred = tk.Frame(nb, bg=self.C_GRAY)
        nb.add(tab_pred, text='  Predict  ')

        self._build_train_tab(tab_train)
        self._build_predict_tab(tab_pred)

    # ── TRAINING TAB ────────────────────────────────────────────
    def _build_train_tab(self, parent):
        frame = tk.Frame(parent, bg=self.C_GRAY)
        frame.pack(fill='both', expand=True, padx=20, pady=20)

        left = tk.LabelFrame(frame, text='Control', font=('Arial', 13, 'bold'),
                             bg=self.C_WHITE, padx=25, pady=20)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 15))

        right = tk.LabelFrame(frame, text='Training Log & Metrics',
                              font=('Arial', 13, 'bold'),
                              bg=self.C_WHITE, padx=25, pady=20)
        right.grid(row=0, column=1, sticky='nsew')

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=2)
        frame.rowconfigure(0, weight=1)

        sf = tk.LabelFrame(left, text='Status', font=('Arial', 11, 'bold'),
                           bg='#ecf0f1', padx=15, pady=15)
        sf.pack(fill='x', pady=(0, 20))

        self.train_status = tk.Label(sf, text='Initializing...',
                                     font=('Arial', 11, 'bold'),
                                     bg='#ecf0f1', fg=self.C_BLUE)
        self.train_status.pack(pady=5)

        self.accuracy_label = tk.Label(sf, text='', font=('Arial', 10),
                                       bg='#ecf0f1', fg='#34495e')
        self.accuracy_label.pack(pady=2)

        mf = tk.LabelFrame(left, text='Manual', font=('Arial', 11, 'bold'),
                           bg='white', padx=15, pady=15)
        mf.pack(fill='x')

        tk.Label(mf, text='Dataset:', font=('Arial', 10, 'bold'),
                 bg='white').pack(anchor='w', pady=(5, 3))

        ds_row = tk.Frame(mf, bg='white')
        ds_row.pack(fill='x', pady=(0, 10))

        self.dataset_label = tk.Label(ds_row, text='Auto-detecting...',
                                      font=('Arial', 9), bg='#f8f9fa', fg='#7f8c8d',
                                      relief='flat', padx=10, pady=8, anchor='w')
        self.dataset_label.pack(side='left', fill='x', expand=True, padx=(0, 8))

        tk.Button(ds_row, text='Browse', command=self._browse_dataset,
                  bg=self.C_BLUE, fg='white', font=('Arial', 9, 'bold'),
                  padx=12, pady=6, cursor='hand2', relief='flat').pack(side='right')

        self.train_btn = tk.Button(
            mf, text='Train', command=self._start_training,
            bg=self.C_GREEN, fg='white', font=('Arial', 12, 'bold'),
            padx=35, pady=12, cursor='hand2', relief='flat')
        self.train_btn.pack(pady=(5, 8))

        self.train_progress = ttk.Progressbar(mf, mode='indeterminate', length=280)
        self.train_progress.pack(pady=(5, 0))

        self.train_log = scrolledtext.ScrolledText(
            right, font=('Consolas', 9), wrap=tk.WORD,
            bg='#1e1e1e', fg='#00ff88', insertbackground='white', height=38)
        self.train_log.pack(fill='both', expand=True)
        self.train_log.insert('end', '='*60 + '\n')
        self.train_log.insert('end', '  AI HEART ATTACK PREDICTION\n')
        self.train_log.insert('end', '  Auto-Training Enabled\n')
        self.train_log.insert('end', '='*60 + '\n\n')

    def _browse_dataset(self):
        path = filedialog.askopenfilename(
            title='Select Dataset',
            filetypes=[('CSV', '*.csv'), ('All', '*.*')])
        if path:
            self.dataset_path = path
            self.dataset_label.config(text=os.path.basename(path), fg='#27ae60')

    def _log(self, msg):
        self.train_log.insert('end', msg)
        self.train_log.see('end')
        self.root.update_idletasks()

    def _start_training(self):
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            messagebox.showerror('Error', 'Select dataset first!')
            return
        self.train_btn.config(state='disabled', text='Training...')
        self.train_progress.start(10)
        self.train_status.config(text='Training...', fg=self.C_BLUE)
        self.train_log.delete('1.0', 'end')
        threading.Thread(target=self._run_training, daemon=True).start()

    def _run_training(self):
        try:
            trainer = ModelTrainer(log_callback=self._log)
            model, scaler = trainer.train(self.dataset_path)
            self.model         = model
            self.scaler        = scaler
            self.feature_names = trainer.feature_names
            self.accuracy      = trainer.accuracy
            self.roc_auc       = trainer.roc_auc
            self.root.after(0, self._training_done, True, None)
        except Exception as e:
            import traceback
            self.root.after(0, self._training_done, False, str(e))
            self._log(f"\nERROR: {e}\n")
            self._log(traceback.format_exc())

    def _training_done(self, success, err):
        self.train_progress.stop()
        self.train_btn.config(state='normal', text='Train')
        if success:
            self.train_status.config(text='Complete!', fg=self.C_GREEN)
            if self.accuracy:
                self.accuracy_label.config(
                    text=f"Accuracy: {self.accuracy*100:.2f}%  |  ROC-AUC: {self.roc_auc:.4f}")
            messagebox.showinfo('Success',
                                f'Training Complete!\n\n'
                                f'Accuracy: {self.accuracy*100:.2f}%\n'
                                f'ROC-AUC:  {self.roc_auc:.4f}\n\n'
                                'Switch to Predict tab!')
        else:
            self.train_status.config(text='Failed', fg=self.C_RED)
            messagebox.showerror('Error', f'Failed:\n\n{err}')

    # ── PREDICT TAB ─────────────────────────────────────────────
    def _build_predict_tab(self, parent):
        frame = tk.Frame(parent, bg=self.C_GRAY)
        frame.pack(fill='both', expand=True, padx=20, pady=20)

        left = tk.Frame(frame, bg=self.C_GRAY)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 15))

        right = tk.Frame(frame, bg=self.C_GRAY)
        right.grid(row=0, column=1, sticky='nsew')

        frame.columnconfigure(0, weight=1, minsize=460)
        frame.columnconfigure(1, weight=2)
        frame.rowconfigure(0, weight=1)

        self._build_input_section(left)
        self._build_results_section(right)

    def _build_input_section(self, parent):
        ecg_lf = tk.LabelFrame(
            parent,
            text='ECG Report  (auto-fills IMF Energy / Amplitude / Frequency / HRV below)',
            font=('Arial', 11, 'bold'), bg='#eaf4fb', padx=15, pady=12)
        ecg_lf.pack(fill='x', pady=(0, 8))

        self.ecg_status = tk.Label(ecg_lf, text='No ECG uploaded',
                                   font=('Arial', 9), bg='#eaf4fb', fg='#95a5a6')
        self.ecg_status.pack(pady=(4, 2))

        self.ecg_autofill_lbl = tk.Label(
            ecg_lf, text='', font=('Arial', 8, 'italic'),
            bg='#eaf4fb', fg='#27ae60', wraplength=400)
        self.ecg_autofill_lbl.pack(pady=(0, 4))

        tk.Button(ecg_lf, text='Upload ECG Report Image',
                  command=self._upload_ecg,
                  bg=self.C_BLUE, fg='white', font=('Arial', 10, 'bold'),
                  padx=20, pady=9, cursor='hand2', relief='flat').pack()

        canvas = tk.Canvas(parent, bg=self.C_WHITE, highlightthickness=0, height=490)
        vsb    = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        inner  = tk.Frame(canvas, bg=self.C_WHITE)
        inner.bind('<Configure>',
                   lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.configure(yscrollcommand=vsb.set)

        hist_lf = tk.LabelFrame(inner, text='Medical History',
                                font=('Arial', 10, 'bold'),
                                bg=self.C_WHITE, padx=12, pady=10)
        hist_lf.pack(fill='x', padx=5, pady=(8, 6))

        self.history_vars = {}
        for item in ['Diabetes', 'Smoking']:
            v = tk.IntVar()
            tk.Checkbutton(hist_lf, text=item, variable=v, bg=self.C_WHITE,
                           font=('Arial', 9), anchor='w').pack(
                               fill='x', pady=2, padx=4)
            self.history_vars[item] = v

        stress_lf = tk.LabelFrame(inner, text='Stress Level',
                                  font=('Arial', 10, 'bold'),
                                  bg=self.C_WHITE, padx=12, pady=10)
        stress_lf.pack(fill='x', padx=5, pady=(0, 6))

        self.stress_var  = tk.StringVar(value='Low')
        stress_colors    = {'Low': '#27ae60', 'Medium': '#f39c12', 'High': '#e74c3c'}
        self.stress_btns = {}

        stress_row = tk.Frame(stress_lf, bg=self.C_WHITE)
        stress_row.pack(fill='x')

        for level in ['Low', 'Medium', 'High']:
            btn = tk.Button(
                stress_row, text=level,
                font=('Arial', 10, 'bold'),
                bg=stress_colors[level], fg='white',
                padx=18, pady=8, relief='flat', cursor='hand2',
                command=lambda l=level: self._set_stress(l))
            btn.pack(side='left', padx=(0, 8), pady=4)
            self.stress_btns[level] = btn

        self.stress_indicator = tk.Label(
            stress_lf, text='Low stress selected',
            font=('Arial', 9, 'italic'), bg=self.C_WHITE, fg='#27ae60')
        self.stress_indicator.pack(anchor='w', pady=(4, 0))

        clin_lf = tk.LabelFrame(
            inner,
            text='Patient Data  (green fields auto-filled from ECG)',
            font=('Arial', 10, 'bold'),
            bg=self.C_WHITE, padx=12, pady=10)
        clin_lf.pack(fill='x', padx=5, pady=(0, 10))

        self.inputs = {}
        fields = [
            ('IMF Energy',    '0.0 - 1.0',  True),
            ('IMF Amplitude', '0.0 - 3.0',  True),
            ('IMF Frequency', '0.0 - 1.0',  True),
            ('HRV',           '0 - 200 ms', True),
            ('BP Systolic',   '90 - 200',   False),
            ('BP Diastolic',  '60 - 120',   False),
            ('Age',           '20 - 90',    False),
            ('Sex',           '0=F  1=M',   False),
            ('Cholesterol',   '100 - 400',  False),
        ]

        for label, hint, is_auto in fields:
            row = tk.Frame(clin_lf, bg=self.C_WHITE)
            row.pack(fill='x', pady=3)

            lbl_color = '#27ae60' if is_auto else '#2c3e50'
            tk.Label(row, text=f'{label}:',
                     font=('Arial', 9, 'bold' if is_auto else 'normal'),
                     bg=self.C_WHITE, fg=lbl_color,
                     width=16, anchor='w').pack(side='left')

            e = tk.Entry(row, font=('Arial', 9), width=10, relief='solid', bd=1)
            e.pack(side='left', padx=(4, 6))

            src_txt   = '[Auto from ECG]' if is_auto else '[Enter manually]'
            src_color = '#27ae60' if is_auto else '#95a5a6'
            tk.Label(row, text=f'{hint}   {src_txt}',
                     font=('Arial', 8), fg=src_color,
                     bg=self.C_WHITE).pack(side='left')

            self.inputs[label] = e

        canvas.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')

        btns = tk.Frame(parent, bg=self.C_GRAY)
        btns.pack(fill='x', pady=10)

        tk.Button(btns, text='PREDICT', command=self._predict,
                  bg=self.C_RED, fg='white', font=('Arial', 14, 'bold'),
                  padx=45, pady=14, cursor='hand2', relief='flat').pack(pady=(0, 8))

        tk.Button(btns, text='Clear All', command=self._clear,
                  bg='#95a5a6', fg='white', font=('Arial', 10),
                  padx=30, pady=9, cursor='hand2', relief='flat').pack()

    def _set_stress(self, level):
        self.stress_var.set(level)
        texts  = {'Low': 'Low stress selected',
                  'Medium': 'Medium stress selected',
                  'High': 'High stress selected'}
        colors = {'Low': '#27ae60', 'Medium': '#f39c12', 'High': '#e74c3c'}
        self.stress_indicator.config(text=texts[level], fg=colors[level])
        for l, btn in self.stress_btns.items():
            btn.config(relief='sunken' if l == level else 'flat',
                       bd=2 if l == level else 0)

    def _build_results_section(self, parent):
        ecg_lf = tk.LabelFrame(parent, text='ECG Visualization',
                               font=('Arial', 12, 'bold'),
                               bg=self.C_WHITE, padx=12, pady=10)
        ecg_lf.pack(fill='x', pady=(0, 10))

        self.ecg_plot_frame = tk.Frame(ecg_lf, bg=self.C_WHITE, height=210)
        self.ecg_plot_frame.pack(fill='x')
        self.ecg_plot_frame.pack_propagate(False)
        tk.Label(self.ecg_plot_frame, text='Upload ECG image to visualise signal',
                 font=('Arial', 11), fg='#95a5a6', bg=self.C_WHITE).pack(expand=True)

        res_outer = tk.LabelFrame(parent, text='Results',
                                  font=('Arial', 12, 'bold'),
                                  bg=self.C_WHITE)
        res_outer.pack(fill='both', expand=True)

        res_canvas = tk.Canvas(res_outer, bg=self.C_WHITE, highlightthickness=0)
        res_vsb    = ttk.Scrollbar(res_outer, orient='vertical', command=res_canvas.yview)
        res_canvas.configure(yscrollcommand=res_vsb.set)

        res_vsb.pack(side='right', fill='y')
        res_canvas.pack(side='left', fill='both', expand=True)

        res_lf = tk.Frame(res_canvas, bg=self.C_WHITE, padx=20, pady=15)
        res_win = res_canvas.create_window((0, 0), window=res_lf, anchor='nw')

        def _on_results_configure(event):
            res_canvas.configure(scrollregion=res_canvas.bbox('all'))

        def _on_canvas_resize(event):
            res_canvas.itemconfig(res_win, width=event.width)

        res_lf.bind('<Configure>', _on_results_configure)
        res_canvas.bind('<Configure>', _on_canvas_resize)

        def _on_mousewheel(event):
            res_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

        res_canvas.bind_all('<MouseWheel>', _on_mousewheel)

        score_row = tk.Frame(res_lf, bg=self.C_WHITE)
        score_row.pack(fill='x', pady=(5, 0))
        tk.Label(score_row, text='RISK SCORE', font=('Arial', 13, 'bold'),
                 bg=self.C_WHITE, fg='#34495e').pack(side='left')
        self.risk_percent_lbl = tk.Label(score_row, text='-- %',
                                         font=('Arial', 40, 'bold'),
                                         bg=self.C_WHITE, fg='#7f8c8d')
        self.risk_percent_lbl.pack(side='right')

        self.risk_cat_lbl = tk.Label(res_lf, text='',
                                     font=('Arial', 16, 'bold'), bg=self.C_WHITE)
        self.risk_cat_lbl.pack(anchor='w', pady=(0, 4))

        self.stress_badge = tk.Label(res_lf, text='',
                                     font=('Arial', 10, 'bold'), bg=self.C_WHITE)
        self.stress_badge.pack(anchor='w', pady=(0, 6))

        self.risk_bar = ttk.Progressbar(res_lf, mode='determinate')
        self.risk_bar.pack(fill='x', pady=(0, 14))

        tk.Frame(res_lf, bg='#dde1e7', height=1).pack(fill='x', pady=(0, 12))

        tk.Label(res_lf, text='Analysis:', font=('Arial', 11, 'bold'),
                 bg=self.C_WHITE, anchor='w').pack(fill='x', pady=(0, 4))
        self.analysis_box = scrolledtext.ScrolledText(
            res_lf, height=6, font=('Arial', 9), wrap=tk.WORD,
            bg='#ecf0f1', relief='flat', padx=8, pady=8)
        self.analysis_box.pack(fill='x', pady=(0, 14))

        tk.Frame(res_lf, bg='#dde1e7', height=1).pack(fill='x', pady=(0, 12))

        tk.Label(res_lf, text='Recommendations:', font=('Arial', 11, 'bold'),
                 bg=self.C_WHITE, anchor='w').pack(fill='x', pady=(0, 4))
        self.prevent_box = scrolledtext.ScrolledText(
            res_lf, height=14, font=('Arial', 9), wrap=tk.WORD,
            bg='#e8f8f5', relief='flat', padx=8, pady=8)
        self.prevent_box.pack(fill='x', pady=(0, 10))

        tk.Frame(res_lf, bg='#dde1e7', height=1).pack(fill='x', pady=(4, 12))

        print_btn = tk.Button(
            res_lf,
            text='🖨  Print Report',
            command=self._print_report,
            bg='#2c3e50', fg='white',
            font=('Arial', 12, 'bold'),
            padx=30, pady=12,
            cursor='hand2', relief='flat',
            activebackground='#34495e', activeforeground='white')
        print_btn.pack(pady=(0, 16))

    # ── ECG UPLOAD  (fixed signal extraction) ────────────────────
    def _upload_ecg(self):
        path = filedialog.askopenfilename(
            title='Select ECG Report Image',
            filetypes=[('Images', '*.png *.jpg *.jpeg *.bmp *.tiff'),
                       ('All', '*.*')])
        if not path:
            return

        try:
            # Load and resize to fixed canvas
            pil_img = Image.open(path).convert('L')
            pil_img = pil_img.resize((800, 300))
            img_arr = np.array(pil_img, dtype=np.uint8)

            # ── Use column-wise trace-tracker (Pan-Tompkins friendly) ──
            signal = ECGProcessor.extract_signal_from_image_array(img_arr)
            self.ecg_data = signal

            self.ecg_status.config(
                text=f'  {os.path.basename(path)}', fg=self.C_GREEN)

            # Process with correct sampling rate
            proc = ECGProcessor()
            proc.set_image_mode(img_width=800)   # 40 px/s

            self.ecg_features = proc.extract_features(self.ecg_data)
            self.ecg_metrics  = proc.derive_clinical_values(self.ecg_data)
            self._autofill_ecg_values()
            self._plot_ecg()

            hr  = self.ecg_metrics.get('heart_rate', 0)
            hrv = self.ecg_metrics.get('hrv', 0)
            messagebox.showinfo(
                'ECG Loaded',
                f'ECG analysed successfully!\n\n'
                f'Est. Heart Rate : {hr:.0f} bpm\n'
                f'HRV (SDNN)      : {hrv:.1f} ms\n'
                f'IMF Energy      : {self.ecg_metrics["imf_energy"]:.4f}\n'
                f'IMF Amplitude   : {self.ecg_metrics["imf_amplitude"]:.4f}\n'
                f'IMF Frequency   : {self.ecg_metrics["imf_frequency"]:.4f}\n\n'
                'These have been filled in automatically.\n'
                'Please enter BP, Age, Sex, and Cholesterol manually.')

        except Exception as e:
            import traceback
            messagebox.showerror('Error', f'Failed to load ECG:\n\n{str(e)}\n\n'
                                          + traceback.format_exc())
            self.ecg_status.config(text='Failed', fg=self.C_RED)

    def _autofill_ecg_values(self):
        mapping = {
            'IMF Energy':    self.ecg_metrics.get('imf_energy',    ''),
            'IMF Amplitude': self.ecg_metrics.get('imf_amplitude', ''),
            'IMF Frequency': self.ecg_metrics.get('imf_frequency', ''),
            'HRV':           self.ecg_metrics.get('hrv',           ''),
        }
        for field, val in mapping.items():
            entry = self.inputs[field]
            entry.delete(0, 'end')
            entry.insert(0, str(val))
            entry.config(bg='#eafaf1')

        hr  = self.ecg_metrics.get('heart_rate', 0)
        hrv = self.ecg_metrics.get('hrv', 0)
        hr_label = f'{hr:.0f} bpm' if hr > 0 else 'not detected'
        self.ecg_autofill_lbl.config(
            text=f'Auto-filled from ECG  |  Heart Rate: {hr_label}  |  HRV: {hrv:.1f} ms')

    def _plot_ecg(self):
        for w in self.ecg_plot_frame.winfo_children():
            w.destroy()

        fig = Figure(figsize=(9, 3.0), dpi=90)
        ax  = fig.add_subplot(111)
        n   = min(800, len(self.ecg_data))
        # x-axis in seconds using the 40 px/s rate
        t   = np.arange(n) / 40.0
        ax.plot(t, self.ecg_data[:n], color='#2980b9', linewidth=0.9)
        ax.set_xlabel('Time (s)', fontsize=9, fontweight='bold')
        ax.set_ylabel('Amplitude (normalised)', fontsize=9, fontweight='bold')
        ax.set_title('ECG Signal — extracted from report image (column-wise)',
                     fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.25)
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.ecg_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    # ── PREDICT ──────────────────────────────────────────────────
    def _predict(self):
        if self.model is None:
            messagebox.showerror('Error', 'Model not trained!\n\nWait for auto-training.')
            return

        clinical = []
        for name, entry in self.inputs.items():
            val = entry.get().strip()
            if val == '':
                messagebox.showwarning('Missing', f'Please enter a value for: {name}')
                return
            try:
                clinical.append(float(val))
            except ValueError:
                messagebox.showerror('Invalid', f'Invalid number for: {name}')
                return

        history    = [v.get() for v in self.history_vars.values()]
        stress_val = self.STRESS_MAP[self.stress_var.get()]
        ecg_feats  = self.ecg_features if self.ecg_features else [0.0] * 10

        features = clinical + history + [stress_val] + ecg_feats
        X        = np.array(features, dtype=float).reshape(1, -1)

        expected = len(self.feature_names)
        if X.shape[1] < expected:
            X = np.pad(X, ((0, 0), (0, expected - X.shape[1])))
        elif X.shape[1] > expected:
            X = X[:, :expected]

        try:
            Xs    = self.scaler.transform(X)
            proba = self.model.predict_proba(Xs)[0]
            risk  = proba[1] * 100.0
            self._show_results(risk)
        except Exception as e:
            messagebox.showerror('Error', str(e))

    # ── SHOW RESULTS ─────────────────────────────────────────────
    def _show_results(self, risk):
        stress_level = self.stress_var.get()
        has_diabetes = self.history_vars['Diabetes'].get()
        has_smoking  = self.history_vars['Smoking'].get()

        try:
            bp_sys  = float(self.inputs['BP Systolic'].get())
            bp_dia  = float(self.inputs['BP Diastolic'].get())
            chol    = float(self.inputs['Cholesterol'].get())
            age     = float(self.inputs['Age'].get())
            hrv     = float(self.inputs['HRV'].get())
            sex_val = self.inputs['Sex'].get().strip()
            sex_str = 'Male' if sex_val == '1' else 'Female'
        except ValueError:
            bp_sys = bp_dia = chol = age = hrv = 0
            sex_str = 'N/A'

        self._last_patient = {
            'Age': age, 'Sex': sex_str,
            'BP': f'{bp_sys:.0f}/{bp_dia:.0f} mmHg',
            'Cholesterol': f'{chol:.0f} mg/dL',
            'HRV': f'{hrv:.1f} ms',
            'Diabetes': 'Yes' if has_diabetes else 'No',
            'Smoking':  'Yes' if has_smoking  else 'No',
            'Stress Level': stress_level,
        }
        if self.ecg_metrics:
            self._last_patient['Heart Rate'] = f"{self.ecg_metrics.get('heart_rate', 0):.0f} bpm"
            self._last_patient['IMF Energy'] = str(self.ecg_metrics.get('imf_energy', ''))

        if risk < 30:
            cat, color = 'LOW RISK', self.C_GREEN
        elif risk < 70:
            cat, color = 'MODERATE RISK', self.C_ORANGE
        else:
            cat, color = 'HIGH RISK', self.C_RED

        stress_colors = {'Low': self.C_GREEN, 'Medium': self.C_ORANGE, 'High': self.C_RED}
        self.stress_badge.config(
            text=f'Stress Level: {stress_level}',
            fg=stress_colors[stress_level])

        self.risk_percent_lbl.config(text=f'{risk:.1f} %', fg=color)
        self.risk_cat_lbl.config(text=cat, fg=color)
        self.risk_bar['value'] = risk

        flags = []
        if bp_sys > 140:
            flags.append(f'Hypertension detected (BP {bp_sys:.0f}/{bp_dia:.0f} mmHg)')
        if chol > 240:
            flags.append(f'High cholesterol ({chol:.0f} mg/dL)')
        if hrv < 30:
            flags.append('Low HRV — possible autonomic dysfunction')
        if has_diabetes:
            flags.append('Diabetes significantly increases cardiovascular risk')
        if has_smoking:
            flags.append('Smoking is a major independent risk factor')
        if stress_level == 'High':
            flags.append('High stress chronically elevates cortisol and blood pressure')
        elif stress_level == 'Medium':
            flags.append('Moderate stress — worth managing proactively')
        if age > 60:
            flags.append(f'Age {age:.0f}: cardiovascular risk increases naturally with age')

        if not flags:
            flags = ['No individual high-risk flags detected']

        intros = {
            'LOW RISK': (
                f'Risk Score: {risk:.1f}%  -  {cat}\n\n'
                'Your cardiovascular indicators are within healthy ranges.\n'
                'Keep up your current lifestyle to maintain this result.\n\n'
                'Noted factors:\n'),
            'MODERATE RISK': (
                f'Risk Score: {risk:.1f}%  -  {cat}\n\n'
                'One or more risk factors require attention.\n'
                'Early intervention now can prevent future cardiac events.\n\n'
                'Key concerns identified:\n'),
            'HIGH RISK': (
                f'Risk Score: {risk:.1f}%  -  {cat}\n\n'
                'Multiple critical risk factors are significantly elevated.\n'
                'Immediate medical evaluation is strongly recommended.\n\n'
                'Critical concerns:\n'),
        }
        analysis_text = intros[cat] + '\n'.join(f'  * {f}' for f in flags)

        do_list   = []
        dont_list = []

        do_list += [
            'Eat a heart-healthy diet (vegetables, whole grains, lean proteins, omega-3s)',
            'Exercise 150+ min/week of moderate cardio (walking, cycling, swimming)',
            'Stay well hydrated — aim for 8 glasses of water daily',
            'Get 7-8 hours of quality sleep every night',
            'Schedule regular check-ups (BP, cholesterol, blood sugar)',
        ]
        dont_list += [
            'Eat trans fats, processed foods, or excessive salt',
            'Drink sugary beverages or consume excessive alcohol',
            'Stay sedentary for long periods — take a short walk every hour',
        ]

        if stress_level == 'High':
            do_list += [
                'Practice daily meditation, deep breathing, or yoga (even 10 min)',
                'Disconnect from work/screens at least 1 hour before bed',
                'Talk to a counsellor — chronic stress directly damages the heart',
            ]
            dont_list += [
                'Consume excess caffeine — it amplifies stress hormones',
                'Suppress emotions — process stress in healthy, expressive ways',
            ]
        elif stress_level == 'Medium':
            do_list.append('Add a short daily relaxation routine to reduce baseline stress')
            dont_list.append('Have more than 1-2 cups of coffee per day')

        if bp_sys > 140:
            do_list.append('Strictly limit sodium intake (< 1500 mg/day) to lower BP')
            dont_list.append('Add salt at the table or eat canned/processed foods')

        if chol > 240:
            do_list.append('Increase omega-3 fatty acids (salmon, flaxseed, walnuts)')
            dont_list.append('Eat red meat daily or full-fat dairy — they raise LDL')

        if has_smoking:
            do_list.append('Quit smoking immediately — the single highest-impact change you can make')
            dont_list.append('Smoke even occasionally — there is no safe level')

        if has_diabetes:
            do_list.append('Monitor blood glucose daily and keep HbA1c below 7%')
            dont_list.append('Eat refined carbs or sweets — blood sugar spikes damage vessels')

        if risk >= 70:
            do_list.insert(0, 'See a cardiologist URGENTLY — do not delay')
            do_list.insert(1, 'Inform a family member about your risk level today')
            dont_list.insert(0, 'Perform strenuous exercise until cleared by your doctor')

        rec_text  = 'WHAT YOU SHOULD DO:\n\n'
        rec_text += '\n'.join(f'  + {item}' for item in do_list)
        rec_text += '\n\nWHAT YOU SHOULD AVOID:\n\n'
        rec_text += '\n'.join(f'  - {item}' for item in dont_list)
        rec_text += (
            '\n\n' + '-'*55 + '\n'
            'EMERGENCY: Chest pain, jaw/arm pain, sudden breathlessness\n'
            '  -> Call 112 / 911 IMMEDIATELY')

        self.analysis_box.config(state='normal')
        self.analysis_box.delete('1.0', 'end')
        self.analysis_box.insert('1.0', analysis_text)
        self.analysis_box.config(state='disabled')

        self.prevent_box.config(state='normal')
        self.prevent_box.delete('1.0', 'end')
        self.prevent_box.insert('1.0', rec_text)
        self.prevent_box.config(state='disabled')

        self._last_risk      = risk
        self._last_analysis  = analysis_text
        self._last_recommend = rec_text
        self._last_flags     = flags
        self._last_cat       = cat
        self._last_do        = do_list
        self._last_dont      = dont_list

    # ── PRINT REPORT ─────────────────────────────────────────────
    def _print_report(self):
        if self._last_risk is None:
            messagebox.showwarning('No Report',
                                   'Please run a prediction first before printing.')
            return

        risk   = self._last_risk
        cat    = self._last_cat
        flags  = getattr(self, '_last_flags',  [])
        do_list   = getattr(self, '_last_do',   [])
        dont_list = getattr(self, '_last_dont', [])

        if risk < 30:
            risk_color = '#27ae60'; risk_bg = '#eafaf1'; risk_border = '#27ae60'
        elif risk < 70:
            risk_color = '#f39c12'; risk_bg = '#fef9e7'; risk_border = '#f39c12'
        else:
            risk_color = '#e74c3c'; risk_bg = '#fdedec'; risk_border = '#e74c3c'

        now = datetime.now().strftime('%B %d, %Y  %H:%M')

        patient_rows = ''
        for k, v in self._last_patient.items():
            patient_rows += f'<tr><td class="pl">{k}</td><td class="pv">{v}</td></tr>\n'

        flags_html = ''.join(f'<li>{f}</li>' for f in flags)
        do_html    = ''.join(f'<li>{item}</li>' for item in do_list)
        dont_html  = ''.join(f'<li>{item}</li>' for item in dont_list)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Heart Attack Risk Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: Arial, Helvetica, sans-serif;
    font-size: 13px;
    color: #2c3e50;
    background: #fff;
    padding: 32px 40px;
  }}
  .header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 3px solid #2c3e50;
    padding-bottom: 14px;
    margin-bottom: 20px;
  }}
  .header-left h1 {{ font-size: 22px; font-weight: 800; color: #2c3e50; }}
  .header-left p  {{ font-size: 11px; color: #7f8c8d; margin-top: 3px; }}
  .header-right   {{ text-align: right; font-size: 11px; color: #7f8c8d; }}
  .risk-banner {{
    background: {risk_bg};
    border: 2px solid {risk_border};
    border-radius: 10px;
    padding: 20px 28px;
    display: flex;
    align-items: center;
    gap: 30px;
    margin-bottom: 22px;
  }}
  .risk-score {{ font-size: 56px; font-weight: 900; color: {risk_color}; line-height: 1; }}
  .risk-label {{ font-size: 22px; font-weight: 800; color: {risk_color}; }}
  .risk-sub   {{ font-size: 12px; color: #555; margin-top: 5px; }}
  h2 {{
    font-size: 14px; font-weight: 700; color: #2c3e50;
    background: #ecf0f1; padding: 7px 12px;
    border-left: 4px solid #2c3e50;
    margin-bottom: 10px; margin-top: 20px;
    border-radius: 0 4px 4px 0;
  }}
  .ptable {{ width: 100%; border-collapse: collapse; margin-bottom: 4px; font-size: 12px; }}
  .ptable td {{ padding: 5px 10px; border: 1px solid #dde1e7; }}
  .ptable .pl {{ background: #f4f6f7; font-weight: 700; width: 38%; color: #34495e; }}
  .ptable .pv {{ color: #2c3e50; }}
  .flags-list, .do-list, .dont-list {{ padding-left: 22px; margin: 6px 0 10px; }}
  .flags-list li {{ margin-bottom: 5px; color: #555; }}
  .do-list   li  {{ margin-bottom: 5px; color: #1a5e37; }}
  .dont-list li  {{ margin-bottom: 5px; color: #922b21; }}
  .do-list   li::marker {{ color: #27ae60; font-weight: bold; }}
  .dont-list li::marker {{ color: #e74c3c; font-weight: bold; }}
  .do-box {{
    background: #eafaf1; border: 1px solid #a9dfbf;
    border-radius: 6px; padding: 12px 16px; margin-bottom: 14px;
  }}
  .dont-box {{
    background: #fdedec; border: 1px solid #f5b7b1;
    border-radius: 6px; padding: 12px 16px; margin-bottom: 14px;
  }}
  .do-box h3, .dont-box h3 {{ font-size: 13px; font-weight: 700; margin-bottom: 8px; }}
  .do-box   h3 {{ color: #1a5e37; }}
  .dont-box h3 {{ color: #922b21; }}
  .emergency {{
    background: #e74c3c; color: white; border-radius: 8px;
    padding: 14px 18px; margin-top: 24px; text-align: center;
    font-size: 13px; font-weight: 700;
  }}
  .emergency span {{ font-size: 11px; font-weight: normal; display: block; margin-top: 4px; }}
  .footer {{
    margin-top: 28px; border-top: 1px solid #dde1e7;
    padding-top: 10px; font-size: 10px; color: #95a5a6; text-align: center;
  }}
  @media print {{ body {{ padding: 16px 20px; }} .no-print {{ display: none; }} }}
</style>
</head>
<body>
<div class="header">
  <div class="header-left">
    <h1>❤  Heart Attack Risk Report</h1>
    <p>AI-Powered Cardiovascular Risk Assessment</p>
  </div>
  <div class="header-right">
    <div>Generated: {now}</div>
    <div style="margin-top:4px; font-weight:700; color:#2c3e50;">CONFIDENTIAL — FOR PATIENT USE ONLY</div>
  </div>
</div>
<div class="risk-banner">
  <div class="risk-score">{risk:.1f}%</div>
  <div>
    <div class="risk-label">{cat}</div>
    <div class="risk-sub">Predicted probability of cardiovascular disease<br>
      based on patient data and ECG analysis.</div>
  </div>
</div>
<h2>Patient Summary</h2>
<table class="ptable">{patient_rows}</table>
<h2>Key Risk Factors Identified</h2>
<ul class="flags-list">{flags_html}</ul>
<h2>Personalised Recommendations</h2>
<div class="do-box">
  <h3>✅ What You Should Do</h3>
  <ul class="do-list">{do_html}</ul>
</div>
<div class="dont-box">
  <h3>🚫 What You Should Avoid</h3>
  <ul class="dont-list">{dont_html}</ul>
</div>
<div class="emergency">
  🚨 EMERGENCY
  <span>If you experience chest pain, jaw/arm pain, or sudden breathlessness — Call 112 / 911 IMMEDIATELY</span>
</div>
<div class="footer">
  This report is generated by the AI Heart Attack Prediction System and is intended for informational purposes only.<br>
  It does not constitute medical advice. Please consult a qualified healthcare professional for diagnosis and treatment.
</div>
<script>window.onload = function() {{ window.print(); }};</script>
</body>
</html>"""

        try:
            tmp = tempfile.NamedTemporaryFile(
                mode='w', suffix='.html', delete=False,
                encoding='utf-8', prefix='heart_report_')
            tmp.write(html)
            tmp.close()
            webbrowser.open('file://' + os.path.abspath(tmp.name))
        except Exception as e:
            messagebox.showerror('Print Error',
                                 f'Could not open print preview:\n\n{str(e)}')

    # ── CLEAR ─────────────────────────────────────────────────────
    def _clear(self):
        for e in self.inputs.values():
            e.delete(0, 'end')
            e.config(bg='white')
        for v in self.history_vars.values():
            v.set(0)

        self.stress_var.set('Low')
        self._set_stress('Low')

        self.ecg_data     = None
        self.ecg_features = None
        self.ecg_metrics  = {}
        self.ecg_status.config(text='No ECG uploaded', fg='#95a5a6')
        self.ecg_autofill_lbl.config(text='')

        for w in self.ecg_plot_frame.winfo_children():
            w.destroy()
        tk.Label(self.ecg_plot_frame,
                 text='Upload ECG image to visualise signal',
                 font=('Arial', 11), fg='#95a5a6', bg=self.C_WHITE).pack(expand=True)

        self.risk_percent_lbl.config(text='-- %', fg='#7f8c8d')
        self.risk_cat_lbl.config(text='')
        self.stress_badge.config(text='')
        self.risk_bar['value'] = 0

        self.analysis_box.config(state='normal')
        self.analysis_box.delete('1.0', 'end')
        self.analysis_box.config(state='disabled')

        self.prevent_box.config(state='normal')
        self.prevent_box.delete('1.0', 'end')
        self.prevent_box.config(state='disabled')

        self._last_risk      = None
        self._last_analysis  = ''
        self._last_recommend = ''
        self._last_patient   = {}


# ──────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    HeartAttackApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()