import os
import wfdb
import numpy as np
from scipy.fftpack import fft
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from tqdm import tqdm

# Öznitelik çıkarım fonksiyonu (bir kanal için)
def extract_features(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    max_val = np.max(signal)
    min_val = np.min(signal)
    energy = np.sum(signal**2)
    
    # FFT sonrası dominant frekans
    freqs = fft(signal)
    freq_magnitudes = np.abs(freqs[:len(freqs)//2])
    dom_freq = np.argmax(freq_magnitudes)
    
    # Spektral entropi
    power_spectrum = freq_magnitudes**2
    power_spectrum /= np.sum(power_spectrum)
    spec_entropy = entropy(power_spectrum)
    
    return [mean, std, max_val, min_val, energy, dom_freq, spec_entropy]

# Verileri oku
data_dir = "PTB-XL"  # kendi veri klasör yolunu buraya yaz
X = []
y = []

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".hea"):
            record_name = os.path.splitext(file)[0]
            try:
                record = wfdb.rdrecord(os.path.join(root, record_name))
                comments = wfdb.rdheader(os.path.join(root, record_name)).comments
                label = 0
                for comment in comments:
                    comment_lower = comment.lower()
                    if "healthy" in comment_lower or "control" in comment_lower or "no abnormality" in comment_lower:
                        label = 0
                        break
                    else:
                        label = 1
                
                features = []
                for channel in range(record.p_signal.shape[1]):  # tüm kanallar
                    channel_signal = record.p_signal[:, channel]
                    features.extend(extract_features(channel_signal))  # her kanal için 7 öznitelik
                
                X.append(features)
                y.append(label)
            except:
                continue

X = np.array(X)
y = np.array(y)

print(f"Toplam örnek sayısı: {len(X)}")
print(f"Özellik vektörü boyutu: {X.shape[1]}")

# Sınıf dengesine bakalım
from collections import Counter
print("Etiket dağılımı:", Counter(y))

# SMOTE uygulama
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("SMOTE sonrası dağılım:", Counter(y_resampled))

# En iyi 20 özelliği seç
selector = SelectKBest(score_func=mutual_info_classif, k=20)
X_selected = selector.fit_transform(X_resampled, y_resampled)
print("Seçilen öznitelik indeksleri:", selector.get_support(indices=True))

# Veriyi ayır
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# Model eğitimi
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Tahminler
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# Sonuçlar
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

# ROC eğrisi
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label='ROC Curve (AUC = %.2f)' % roc_auc_score(y_test, y_prob))
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

