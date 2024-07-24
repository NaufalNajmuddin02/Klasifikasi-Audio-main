import os
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

app = Flask(__name__)

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
            chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
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
            mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
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

class_labels = {0: "Blues", 1: "Kucing", 2: "Burung"}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        
        predicted_class = predict_audio(file_path)
        return jsonify({"predicted_class": class_labels[predicted_class]})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('runs'):
        os.makedirs('runs')
    app.run(host='192.168.59.249', port=5000)

# # Menjalankan Flask app
# if __name__ == '__main__':
#     app.run(debug=True)
