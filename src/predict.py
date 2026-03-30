import pickle
from feature_extraction import extract_features


def predict_emotion(file):
    model = pickle.load("model.pkl","wb")

    features = extract_features(file)
    features = features.reshape(1,-1)

    return model.predict(features)[0]