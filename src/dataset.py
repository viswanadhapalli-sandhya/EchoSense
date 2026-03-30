# importing the feature extraction and emotion label functions
from feature_extraction import extract_features
from emotion_labels import extract_emotion

import numpy as np
from glob import glob

def load_dataset():
    # importing all audio files from RAVDESS dataset
    audio_files = glob('./data/RAVDESS/*/*.wav')

    X = []
    y = []
    print("Total audio files",len(audio_files))
    for file in audio_files:
        feature = extract_features(file)
        label = extract_emotion(file)

        X.append(feature)
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    return X,y
