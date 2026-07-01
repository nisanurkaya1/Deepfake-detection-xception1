# ================================================
# FULL FRAME ÇIKARMA — FaceForensics++
# Yüz tespiti yok, tüm kare kullanılır
# Real: 20 kare, Fake (DF+F2F): 10 kare → 1:1 oran
# Seed=42, 250 video, 200/25/25 bölme
# EfficientNetB0 için: IMG_SIZE = (224, 224)
# Xception için: IMG_SIZE = (299, 299)
# ================================================

import cv2
import os
import random
from tqdm import tqdm

# =====================
# YOLLAR
# =====================
real_src    = '/content/drive/MyDrive/FaceForensics_Data/original_sequences/youtube/c23/videos'
df_src      = '/content/drive/MyDrive/FaceForensics_Data/manipulated_sequences/Deepfakes/c23/videos'
f2f_src     = '/content/drive/MyDrive/FaceForensics_Data/manipulated_sequences/Face2Face/c23/videos'
SAVE_BASE   = '/content/drive/MyDrive/FaceForensics_Data/Xception_Frames'

# =====================
# AYARLAR
# =====================
IMG_SIZE      = (299, 299)   # Xception için. EfficientNetB0: (224, 224)
N_FRAMES_REAL = 20           # real: 200×20 = 4000 kare
N_FRAMES_FAKE = 10           # fake: 400×10 = 4000 kare

for split in ['train', 'val', 'test']:
    for label in ['real', 'fake']:
        os.makedirs(f'{SAVE_BASE}/{split}/{label}', exist_ok=True)

# =====================
# VİDEO BAZLI BÖLME (seed=42)
# =====================
random.seed(42)
all_ids = sorted([v.replace('.mp4','') for v in os.listdir(real_src) if v.endswith('.mp4')])[:250]
random.shuffle(all_ids)

train_ids = set(all_ids[:200])
val_ids   = set(all_ids[200:225])
test_ids  = set(all_ids[225:])
print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

def get_split(v_id):
    if v_id in train_ids: return 'train'
    if v_id in val_ids:   return 'val'
    return 'test'

# =====================
# KARE ÇIKARMA
# =====================
def isle_video(v_yolu, save_path, prefix, n_frame):
    cap = cv2.VideoCapture(v_yolu)
    if not cap.isOpened(): return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return 0

    indices = [int(i * total / n_frame) for i in range(n_frame)]
    saved = 0

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue
        out = cv2.resize(frame, IMG_SIZE)
        cv2.imwrite(os.path.join(save_path, f'{prefix}_{i:02d}.jpg'), out)
        saved += 1

    cap.release()
    return saved

# =====================
# ANA DÖNGÜ
# =====================
print("\n🚀 Full Frame kare çıkarma başlıyor...")
toplam_kare = 0

for v_id in tqdm(sorted(all_ids)):
    split = get_split(v_id)

    # Real → 20 kare
    real_video = os.path.join(real_src, f"{v_id}.mp4")
    if os.path.exists(real_video):
        toplam_kare += isle_video(real_video,
                                   f'{SAVE_BASE}/{split}/real', v_id, N_FRAMES_REAL)

    # DeepFakes → 10 kare
    for dosya in os.listdir(df_src):
        if dosya.startswith(f"{v_id}_"):
            toplam_kare += isle_video(os.path.join(df_src, dosya),
                                       f'{SAVE_BASE}/{split}/fake', f"df_{v_id}", N_FRAMES_FAKE)
            break

    # Face2Face → 10 kare
    for dosya in os.listdir(f2f_src):
        if dosya.startswith(f"{v_id}_"):
            toplam_kare += isle_video(os.path.join(f2f_src, dosya),
                                       f'{SAVE_BASE}/{split}/fake', f"f2f_{v_id}", N_FRAMES_FAKE)
            break

print(f"\n✅ Tamamlandı! Toplam kare: {toplam_kare}")
for split in ['train', 'val', 'test']:
    real_n = len(os.listdir(f'{SAVE_BASE}/{split}/real'))
    fake_n = len(os.listdir(f'{SAVE_BASE}/{split}/fake'))
    print(f"{split}: real={real_n}, fake={fake_n}, oran={fake_n/real_n:.2f}x")
