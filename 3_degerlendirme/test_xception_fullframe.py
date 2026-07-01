# ================================================
# TEST — Xception + LSTM | Full Frame Senaryosu
# Video bazlı çoğunluk oylaması: 1, 5, 10, 50 kare
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
from tensorflow.keras.applications.xception import preprocess_input

MODEL_PATH = '/content/drive/MyDrive/FaceForensics_Data/xception_lstm_fullframe.keras'
real_src   = '/content/drive/MyDrive/FaceForensics_Data/original_sequences/youtube/c23/videos'
df_src     = '/content/drive/MyDrive/FaceForensics_Data/manipulated_sequences/Deepfakes/c23/videos'
f2f_src    = '/content/drive/MyDrive/FaceForensics_Data/manipulated_sequences/Face2Face/c23/videos'
DATA_PATH  = '/content/drive/MyDrive/FaceForensics_Data/Xception_Frames'
SAVE_PATH  = '/content/drive/MyDrive/FaceForensics_Data/Grafikler_Xception_LSTM_FullFrame'
os.makedirs(SAVE_PATH, exist_ok=True)

IMG_SIZE       = (299, 299)
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
        if parca[0] == 'df':
            vid = f"df_{parca[1]}"
        elif parca[0] == 'f2f':
            vid = f"f2f_{parca[1]}"
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
    print(f"  Kare sayısı: {n_frame} — Xception+LSTM Full Frame")
    print(f"{'='*50}")
    y_true, y_pred, y_prob = [], [], []

    for v_id, vid_id, gercek, kaynak in test_videolari:
        label_klasor = 'real' if gercek == 1 else 'fake'
        frames = load_frames(vid_id, label_klasor, n_frame)
        if
