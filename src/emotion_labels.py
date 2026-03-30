# import necessary libraries
import os

emotions = ["Neutral","Calm","Happy","Sad","Angry","Fearful","Disgust","Surprised"]

def extract_emotion(file):
    filename = os.path.basename(file)
    parts = filename.split("-")
    match(parts[2]):
        case "01": 
            return emotions[0]
        case "02": 
            return emotions[1]
        case "03": 
            return emotions[2]
        case "04": 
            return emotions[3]
        case "05": 
            return emotions[4]
        case "06": 
            return emotions[5]
        case "07": 
            return emotions[6]
        case "08": 
            return emotions[7]
