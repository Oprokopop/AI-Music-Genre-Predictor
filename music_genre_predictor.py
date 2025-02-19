import librosa
import numpy as np

def extract_features(audio_path):
    """Extracts MFCC features from an audio file."""
    y, sr = librosa.load(audio_path, duration=30)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = np.mean(mfccs.T, axis=0)
    return features

if __name__ == "__main__":
    audio_file = input("Enter the path to your music file: ")
    features = extract_features(audio_file)
    print("Extracted Features:", features)
    # Note: Integrate a pre-trained classifier here to predict genre based on features.
