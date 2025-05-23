import os
import numpy as np
import wfdb
from scipy.fft import rfft, rfftfreq
from scipy.stats import entropy
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# -------------------------------------
# PARAMETRELER
# -------------------------------------
sampling_rate = 500
data_path = 'C:/Users/merve/OneDrive/Masaüstü/PythonProjects/diagnostic_ecg/data/my_ecg_data'
n_channels = 12

# -------------------------------------
# ÖZNİTELİK ÇIKARIMI (12 KANAL)
# -------------------------------------
def extract_features_multi_channel(signal_matrix, n_channels=12):
    feats = []
    for ch in range(min(n_channels, signal_matrix.shape[1])):
        x = signal_matrix[:, ch]
        # Temel istatistikler
        feats += [
            np.mean(x),
            np.std(x),
            np.max(x) - np.min(x),               # range
            np.mean(np.abs(x - np.mean(x))),     # mean absolute deviation
            np.sum(x**2),                        # energy
        ]
        # FFT
        freqs = rfftfreq(len(x), d=1/sampling_rate)
        fftv = np.abs(rfft(x))
        fftn = fftv / fftv.sum() if fftv.sum() > 0 else fftv
        feats += [
            freqs[np.argmax(fftv)],              # dominant frequency
            entropy(fftn),                       # spectral entropy
            np.sum((fftv > np.mean(fftv)).astype(int)),  # count of peaks above mean
        ]
    return feats

# -------------------------------------
# VERİYİ OKU VE ETİKETLE
# -------------------------------------
X, y = [], []

for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    if not os.path.isdir(folder_path):
        continue
    for fname in os.listdir(folder_path):
        if not fname.endswith('.hea'):
            continue
        record_path = os.path.join(folder_path, fname[:-4])
        try:
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal
            if signal is None or signal.shape[1] < n_channels:
                continue

            header = wfdb.rdheader(record_path)
            keywords = ['diagnosis', 'diagnoses', 'reason', 'indication', 'clinical']
            comments = [c.lower() for c in header.comments]
            diag_lines = [c for c in comments if any(k in c for k in keywords)]
            if not diag_lines:
                continue
            text = ' '.join(diag_lines)
            label = 0 if any(w in text for w in ['healthy', 'no abnormality', 'control']) else 1

            feats = extract_features_multi_channel(signal, n_channels=n_channels)
            X.append(feats)
            y.append(label)
        except Exception:
            continue

X = np.array(X)
y = np.array(y)

# -------------------------------------
# TRAIN/TEST SPLIT
# -------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------
# PIPELINE + GRID SEARCH
# -------------------------------------
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scale', StandardScaler()),
    ('select', SelectKBest(score_func=mutual_info_classif)),
    ('clf', RandomForestClassifier(random_state=42, n_jobs=-1)),
])

param_grid = {
    'smote__k_neighbors':     [3, 5, 7],
    'select__k':              [8, 12, 16, 'all'],
    'clf__n_estimators':      [100, 200, 300],
    'clf__max_depth':         [None, 10, 20],
    'clf__min_samples_split': [2, 5, 10],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)
search.fit(X_train, y_train)

print("=== En İyi Parametreler ===")
print(search.best_params_)
print(f"CV Accuracy: {search.best_score_:.4f}")

# -------------------------------------
# FINAL DEĞERLENDİRME
# -------------------------------------
best_pipe = search.best_estimator_
y_pred = best_pipe.predict(X_test)
y_prob = best_pipe.predict_proba(X_test)[:, 1]

print("\n=== Test Set Metrics ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC  : {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc_score(y_test, y_prob):.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Healthy (0)', 'Patient (1)'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()
