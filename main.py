import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    with tf.device('/cpu:0'):
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        features = []
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
            features.append(mfccs)
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T,axis=0)
            features.append(chroma)
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T,axis=0)
            features.append(mel)
    return np.concatenate(features)

data_blues = "Dataset\Blues"
data_kucing = "Dataset\Kucing"
data_burung = "Dataset\Burung"

features = []
labels = []

for file in os.listdir(data_blues):
    file_path = os.path.join(data_blues, file)
    feature = extract_features(file_path)
    features.append(feature)
    labels.append(0)  

for file in os.listdir(data_kucing):
    file_path = os.path.join(data_kucing, file)
    feature = extract_features(file_path)
    features.append(feature)
    labels.append(1)

for file in os.listdir(data_burung):
    file_path = os.path.join(data_burung, file)
    feature = extract_features(file_path)
    features.append(feature)
    labels.append(2) 

features = np.array(features)
labels = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Reshape((X_train.shape[1], 1)),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(256, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)

loss, accuracy = model.evaluate(X_val, y_val)
print("Validation Accuracy:", accuracy)

model.summary()

model_save_path = "Model"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
model.save(os.path.join(model_save_path, "audio_classification_model.h5"))
