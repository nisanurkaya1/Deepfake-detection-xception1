# ================================================
# TEST — EfficientNetB0 + LSTM | Face Crop Senaryosu
# Video bazlı çoğunluk oylaması: 1, 5, 10, 50 kare
# En iyi sonuç: %84,00 doğruluk, AUC=0,9040 (50 kare)
# ================================================

import tensorflow as tf
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, accuracy_score,
                              f1_score, precision_score, recall_score)
from tensorflow.keras.applications.efficientnet import preprocess_input

MODEL_PATH = '/content/drive/MyDrive/FaceForensics_Data/efficientnet_lstm_facecrop.keras'
real_src   = '/content/drive/MyDrive/FaceForensics_Data/original_sequences/youtube/c23/videos'
df_src     = '/content/drive/MyDrive/FaceForensics_Data/manipulated_sequences/Deepfakes/c23/videos'
f2f_src    = '/content/drive/MyDrive/FaceForensics_Data/manipulated_sequences/Face2Face/c23/videos'
DATA_PATH  = '/content/drive/MyDrive/FaceForensics_Data/Xception_Frames_FaceCrop'
SAVE_PATH  = '/content/drive/MyDrive/FaceForensics_Data/Grafikler_EfficientNet_LSTM'
os.makedirs(SAVE_PATH, exist_ok=True)

IMG_SIZE       = (224, 224)
N_FRAMES_MODEL = 10

random.seed(42)
all_ids = sorted([v.replace('.mp4','') for v in os.listdir(real_src) if v.endswith('.mp4')])[:250]
random.shuffle(all_ids)
test_ids = set(all_ids[225:])
print(f"Test video sayısı: {len(test_ids)}")

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model yüklendi!")

def load_frames(video_id, label_klasor, n_frame):
    klasor = os.path.join(DATA_PATH, 'test', label_klasor)
    video_dict = {}
    for dosya in sorted(os.listdir(klasor)):
        if not dosya.endswith('.jpg'):
            continue
        parca = dosya.replace('.jpg', '').split('_')
        if parca[0] == 'REAL':
            vid = parca[1]
        elif parca[0] in ['DF', 'F2F']:
            vid = f"{parca[0]}_{parca[1]}"
        else:
            vid = parca[0]
        if vid not in video_dict:
            video_dict[vid] = []
        video_dict[vid].append(os.path.join(klasor, dosya))

    if video_id not in video_dict:
        return None

    frame_paths = sorted(video_dict[video_id])
    toplam = len(frame_paths)
    if toplam >= n_frame:
        secilen = [frame_paths[int(i*toplam/n_frame)] for i in range(n_frame)]
    else:
        secilen = frame_paths
        while len(secilen) < n_frame:
            secilen.append(secilen[-1])

    frames = []
    for path in secilen:
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        frames.append(img.numpy())

    frames = np.array(frames)
    if frames.shape[0] < N_FRAMES_MODEL:
        tekrar = np.tile(frames, (N_FRAMES_MODEL // frames.shape[0] + 1, 1, 1, 1))
        frames = tekrar[:N_FRAMES_MODEL]
    return frames[:N_FRAMES_MODEL]

test_videolari = []
for v_id in sorted(test_ids):
    if os.path.exists(os.path.join(real_src, f"{v_id}.mp4")):
        test_videolari.append((v_id, v_id, 1, 'real'))
    for dosya in sorted(os.listdir(df_src)):
        if dosya.startswith(f"{v_id}_"):
            test_videolari.append((v_id, f"df_{v_id}", 0, 'deepfakes'))
            break
    for dosya in sorted(os.listdir(f2f_src)):
        if dosya.startswith(f"{v_id}_"):
            test_videolari.append((v_id, f"f2f_{v_id}", 0, 'face2face'))
            break

print(f"Toplam test videosu: {len(test_videolari)}")

frame_sayilari = [1, 5, 10, 50]
sonuclar = {}

for n_frame in frame_sayilari:
    print(f"\n{'='*50}")
    print(f"  Kare sayısı: {n_frame} — EfficientNetB0+LSTM Face Crop")
    print(f"{'='*50}")
    y_true, y_pred, y_prob = [], [], []

    for v_id, vid_id, gercek, kaynak in test_videolari:
        label_klasor = 'real' if gercek == 1 else 'fake'
        frames = load_frames(vid_id, label_klasor, n_frame)
        if frames is None:
            print(f"  ⚠️  [{kaynak:<10}] {v_id} → Kare bulunamadı")
            continue
        frames_input = np.expand_dims(frames, axis=0)
        prob = model.predict(frames_input, verbose=0)[0][0]
        karar = 1 if prob > 0.5 else 0
        y_true.append(gercek); y_pred.append(karar); y_prob.append(prob)
        dogru = "✅" if karar == gercek else "❌"
        print(f"  {dogru} [{kaynak:<10}] {v_id} → {'REAL' if gercek==1 else 'FAKE'} → {'REAL' if karar==1 else 'FAKE'} (p={prob:.3f})")

    y_true = np.array(y_true); y_pred = np.array(y_pred); y_prob = np.array(y_prob)
    acc  = accuracy_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)
    f1   = f1_score(y_true, y_pred, average='macro')
    prec = precision_score(y_true, y_pred, average='macro')
    rec  = recall_score(y_true, y_pred, average='macro')
    print(f"\n  Doğruluk: %{acc*100:.2f} | AUC: {auc:.4f} | F1: {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=['Fake','Real'], digits=4))
    sonuclar[n_frame] = {'accuracy':acc,'auc':auc,'f1':f1,'precision':prec,
                         'recall':rec,'y_true':y_true,'y_pred':y_pred,'y_prob':y_prob}

best_n = max(sonuclar, key=lambda n: sonuclar[n]['accuracy'])
print(f"\n⭐ En iyi kare sayısı: {best_n} → %{sonuclar[best_n]['accuracy']*100:.2f}")

cm = confusion_matrix(sonuclar[10]['y_true'], sonuclar[10]['y_pred'])
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake','Real'], yticklabels=['Fake','Real'])
plt.title('Karmaşıklık Matrisi — EfficientNetB0+LSTM Face Crop (10 Kare)', fontsize=12)
plt.xlabel('Tahmin Edilen Sınıf'); plt.ylabel('Gerçek Sınıf')
plt.tight_layout()
plt.savefig(f'{SAVE_PATH}/confusion_matrix_video.png', dpi=150); plt.show()

fpr, tpr, _ = roc_curve(sonuclar[10]['y_true'], sonuclar[10]['y_prob'])
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC={sonuclar[10]["auc"]:.4f})')
plt.plot([0,1],[0,1], color='navy', lw=1, linestyle='--')
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('ROC Eğrisi — EfficientNetB0+LSTM Face Crop (10 Kare)', fontsize=12)
plt.legend(loc='lower right'); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{SAVE_PATH}/roc_curve_video.png', dpi=150); plt.show()

print(f"\n{'='*65}")
print(f"ÖZET — EfficientNetB0+LSTM Face Crop Video Bazlı")
print(f"{'='*65}")
print(f"{'Kare':<8} | {'Doğruluk':<10} | {'F1':<8} | {'Kesinlik':<10} | {'Duyarlılık':<8} | {'AUC':<8}")
print(f"{'-'*65}")
for n in frame_sayilari:
    s = sonuclar[n]
    print(f"{n:<8} | %{s['accuracy']*100:<8.2f} | {s['f1']:<8.4f} | {s['precision']:<10.4f} | {s['recall']:<8.4f} | {s['auc']:.4f}")

print("\n✅ Tüm testler tamamlandı!")
