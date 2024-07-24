import os
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    with tf.device('/cpu:0'):
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        features = []
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
            features.append(mfccs)
            # Plot and save the MFCC
            plt.figure(figsize=(10, 4))
            plt.imshow(mfccs.reshape(-1, 1), cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title('MFCC')
            plt.tight_layout()
            plt.savefig('runs/MFCC_{}.png'.format(os.path.basename(file_path)))
            plt.close()
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T,axis=0)
            features.append(chroma)
            # Plot and save the Chroma
            plt.figure(figsize=(10, 4))
            plt.imshow(chroma.reshape(-1, 1), cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title('Chroma')
            plt.tight_layout()
            plt.savefig('runs/Chroma_{}.png'.format(os.path.basename(file_path)))
            plt.close()
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T,axis=0)
            features.append(mel)
            # Plot and save the Mel spectrogram
            plt.figure(figsize=(10, 4))
            plt.imshow(mel.reshape(-1, 1), cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title('Mel Spectrogram')
            plt.tight_layout()
            plt.savefig('runs/Mel_{}.png'.format(os.path.basename(file_path)))
            plt.close()
    return np.concatenate(features)

model_path = "Model/audio_classification_model.h5"
model = tf.keras.models.load_model(model_path)

def predict_audio(file_path):
    feature = extract_features(file_path)
    prediction = model.predict(np.expand_dims(feature, axis=0))
    
    predicted_class = np.argmax(prediction)
    
    return predicted_class

file_path_to_predict = "Test/blues.wav"

predicted_class = predict_audio(file_path_to_predict)

class_labels = {0: "Blues", 1: "Kucing", 2: "Burung"}

print("Predicted Class:", class_labels[predicted_class])
