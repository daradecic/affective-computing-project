import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


class TFEmotionClassifier:
    def __init__(self):
        self.model = load_model('models/7CLASSFERModel.h5')
        
    def classify(self, img_path):
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (48, 48))
        img = image.img_to_array(img)
        img = img.reshape(1, 2304)
        rescaled = []
        for val in img[0]:
            rescaled.append(val / 255.0)
        rescaled = np.array(rescaled)
        img = rescaled.reshape(48, 48)
        img = img.reshape(1, 48, 48, 1)
        classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Neutral', 'Surprise']
        preds = self.model.predict(img)
        probabilities = []
        for i, item in enumerate(preds[0]):
            probabilities.append({
                'Class': classes[i],
                'Probability': item
            })
        probabilities = pd.DataFrame(probabilities)
        probabilities = probabilities.sort_values(by='Probability', ascending=False)
        return classes[preds.argmax()], probabilities