# ================================================
# EĞİTİM — Xception + LSTM | Full Frame Senaryosu
# FaceForensics++ DeepFakes + Face2Face (c23)
# Video bazlı bölme, seed=42
# ================================================

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import json
from tensorflow.keras import layers
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input

IMG_SIZE  = (299, 299)
N_FRAMES  = 10
BATCH     = 8
AUTOTUNE  = tf.data.AUTOTUNE

DATA_PATH  = '/content/drive/MyDrive/FaceForensics_Data/Xception_Frames'
MODEL_PATH = '/content/drive/MyDrive/FaceForensics_Data/xception_lstm_fullframe.keras'
SAVE_PATH  = '/content/drive/MyDrive/FaceForensics_Data/Grafikler_Xception_LSTM_FullFrame'
os.makedirs(SAVE_PATH, exist_ok=True)

def video_ds_olustur(split):
    all_paths, all_labels = [], []
    for label_name, label_val in [('fake', 0), ('real', 1)]:
        klasor = os.path.join(DATA_PATH, split, label_name)
        video_dict = {}
        for dosya in sorted(os.listdir(klasor)):
            if not dosya.endswith('.jpg'):
                continue
            parca = dosya.replace('.jpg', '').split('_')
            if parca[0] == 'df':
                video_id = f"df_{parca[1]}"
            elif parca[0] == 'f2f':
                video_id = f"f2f_{parca[1]}"
            else:
                video_id = parca[0]
            if video_id not in video_dict:
                video_dict[video_id] = []
            video_dict[video_id].append(os.path.join(klasor, dosya))
        for video_id, frame_paths in video_dict.items():
            frame_paths = sorted(frame_paths)
            if len(frame_paths) >= N_FRAMES:
                secilen = frame_paths[::max(1, len(frame_paths)//N_FRAMES)][:N_FRAMES]
            else:
                secilen = frame_paths
                while len(secilen) < N_FRAMES:
                    secilen.append(secilen[-1])
            all_paths.append(secilen)
            all_labels.append(label_val)
    real_n = sum(all_labels)
    fake_n = len(all_labels) - real_n
    print(f"{split}: {len(all_paths)} video | real={real_n}, fake={fake_n}")
    return all_paths, all_labels

train_paths, train_labels = video_ds_olustur('train')
val_paths,   val_labels   = video_ds_olustur('val')

def load_video_frames(frame_paths):
    frames = []
    for path in frame_paths:
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        frames.append(img)
    return tf.stack(frames)

def generator(paths, labels):
    for frame_paths, label in zip(paths, labels):
        frames = load_video_frames(frame_paths)
        yield frames, tf.cast(label, tf.float32)

def tf_dataset(paths, labels, shuffle=True, repeat=False):
    ds = tf.data.Dataset.from_generator(
        lambda: generator(paths, labels),
        output_signature=(
            tf.TensorSpec(shape=(N_FRAMES, 299, 299, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )
    if shuffle:
        ds = ds.shuffle(500, seed=42)
    if repeat:
        ds = ds.repeat()
    return ds.batch(BATCH).prefetch(AUTOTUNE)

train_ds = tf_dataset(train_paths, train_labels, shuffle=True,  repeat=True)
val_ds   = tf_dataset(val_paths,   val_labels,   shuffle=False, repeat=False)
print("✅ Dataset hazır!")

steps_per_epoch  = len(train_paths) // BATCH
validation_steps = len(val_paths)   // BATCH

base = Xception(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
base.trainable = True
for layer in base.layers[:-80]:
    layer.trainable = False

frame_input = tf.keras.Input(shape=(299, 299, 3))
fx = base(frame_input, training=True)
fx = layers.GlobalAveragePooling2D()(fx)
fx = layers.BatchNormalization()(fx)
cnn_model = tf.keras.Model(frame_input, fx)

video_input = tf.keras.Input(shape=(N_FRAMES, 299, 299, 3))
x = layers.TimeDistributed(cnn_model)(video_input)
x = layers.LSTM(128, return_sequences=False)(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(video_input, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(2.74e-05),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15,
        restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2,
        patience=5, min_lr=1e-7, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor='val_loss',
        save_best_only=True, verbose=1)
]

print("\n🚀 Xception+LSTM Full Frame eğitim başlıyor...")
history = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    epochs=100,
    callbacks=callbacks,
    verbose=1
)

history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open(f'{SAVE_PATH}/history.json', 'w') as f:
    json.dump(history_dict, f)

epochs = range(1, len(history.history['loss']) + 1)

plt.figure(figsize=(10, 4))
plt.plot(epochs, history.history['loss'],     'b-o', markersize=4, label='Eğitim Kaybı')
plt.plot(epochs, history.history['val_loss'], 'r-o', markersize=4, label='Doğrulama Kaybı')
plt.title('Epoch vs. Kayıp — Xception+LSTM Full Frame', fontsize=14)
plt.xlabel('Epoch'); plt.ylabel('Kayıp')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(f'{SAVE_PATH}/loss_grafigi.png', dpi=150); plt.close()

plt.figure(figsize=(10, 4))
plt.plot(epochs, history.history['accuracy'],     'b-o', markersize=4, label='Eğitim')
plt.plot(epochs, history.history['val_accuracy'], 'r-o', markersize=4, label='Doğrulama')
plt.title('Epoch vs. Doğruluk — Xception+LSTM Full Frame', fontsize=14)
plt.xlabel('Epoch'); plt.ylabel('Doğruluk')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(f'{SAVE_PATH}/accuracy_grafigi.png', dpi=150); plt.close()

print(f"✅ Tamamlandı! Grafikler: {SAVE_PATH}")
