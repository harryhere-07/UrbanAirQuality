import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class AQIModel:
    def __init__(self):
        self.model = self._build_model()
        self.labels = ["Good", "Moderate", "Unhealthy", "Hazardous"]

    def _build_model(self):
        """
        Builds a standard CNN for image classification.
        """
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(4, activation='softmax') # 4 Categories of AQI
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def predict_aqi(self, image, calculated_haze_score):
        """
        HYBRID INFERENCE:
        Normally, we would run: prediction = self.model.predict(image)
        However, since the user (you) likely doesn't have a trained .h5 weights file yet,
        we will use the 'calculated_haze_score' from the physics engine to simulate
        the classification logic.
        """
        
        # Logic to simulate Deep Learning classification based on optical density
        if calculated_haze_score < 20:
            category = "Good"
            pm25 = np.random.uniform(0, 12)
        elif calculated_haze_score < 45:
            category = "Moderate"
            pm25 = np.random.uniform(12, 35)
        elif calculated_haze_score < 70:
            category = "Unhealthy"
            pm25 = np.random.uniform(35, 55)
        else:
            category = "Hazardous"
            pm25 = np.random.uniform(55, 150)
            
        return category, pm25