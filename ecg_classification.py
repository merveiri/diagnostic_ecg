import os
import numpy as np
import wfdb
from scipy.fft import rfft, rfftfreq
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Frekans temelli öznitelikler için parametre
sampling_rate = 500  # PTB veri setinde ECG genelde 500 Hz

def extract_features(channel):
    # Temel istatistiksel öznitelikler
    mean_val = np.mean(channel)
    std_val = np.std(channel)
    max_val = np.max(channel)
    min_val = np.min(channel)
    energy = np.sum(channel ** 2)

    # Frekans temelli öznitelikler
    freqs = rfftfreq(len(channel), d=1/sampling_rate)
    fft_vals = np.abs(rfft(channel))
    fft_norm = fft_vals / np.sum(fft_vals)

    dominant_freq = freqs[np.argmax(fft_vals)]
    spectral_entropy = entropy(fft_norm)

    return [mean_val, std_val, max_val, min_val, energy, dominant_freq, spectral_entropy]

# ---------------------------------------------
# VERİYİ OKU, ÖZELLİKLERİ ÇIKAR, ETİKETLE
# ---------------------------------------------

data_path = 'C:/Users/merve/OneDrive/Masaüstü/PythonProjects/diagnostic_ecg/data/my_ecg_data'
X = []
y = []

for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    if not os.path.isdir(folder_path):
        continue
    for file in os.listdir(folder_path):
        if file.endswith('.hea'):
            record_path = os.path.join(folder_path, file.replace('.hea', ''))
            try:
                record = wfdb.rdrecord(record_path)
                signal = record.p_signal
                if signal is None:
                    continue

                header = wfdb.rdheader(record_path)
                keywords = ['diagnosis', 'diagnoses', 'reason', 'indication', 'clinical']
                diagnosis_lines = [c.lower() for c in header.comments if any(k in c.lower() for k in keywords)]
                if not diagnosis_lines:
                    continue
                diagnosis_text = ' '.join(diagnosis_lines)
                label = 0 if 'healthy' in diagnosis_text or 'no abnormality' in diagnosis_text or 'control' in diagnosis_text else 1

                channel = signal[:, 0]
                features = extract_features(channel)
                X.append(features)
                y.append(label)
            except:
                continue

X = np.array(X)
y = np.array(y)

# ----------------------
# ETİKETLER VE SMOTE
# ----------------------

unique, counts = np.unique(y, return_counts=True)
print(f"Sınıflar: {unique}, Adetleri: {counts}")
if len(unique) < 2:
    print("Yeterli sınıf yok.")
    exit()

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print(f"\nSMOTE sonrası örnek sayısı: {len(y_resampled)}")

# -----------------------------------
# ÖZNİTELİK SEÇİMİ VE MODEL EĞİTİMİ
# -----------------------------------

all_feature_names = ['mean', 'std', 'max', 'min', 'energy', 'dominant_freq', 'spectral_entropy']
selector = SelectKBest(score_func=mutual_info_classif, k=4)
X_selected = selector.fit_transform(X_resampled, y_resampled)
selected_indices = selector.get_support(indices=True)

print("\n📌 Seçilen Özellikler:")
for i in selected_indices:
    print(f"- {all_feature_names[i]} (skor: {selector.scores_[i]:.4f})")

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# ----------------------------
# PERFORMANS METRİKLERİ
# ----------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n📊 Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))
print("🔍 Özet Metrikler:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")

# ----------------------------
# ROC VE CONFUSION MATRIX
# ----------------------------

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}', color='darkorange')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy (0)', 'Patient (1)'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()
