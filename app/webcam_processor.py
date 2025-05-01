import numpy as np
from app.age_predictor import AgePredictor

class WebcamProcessor:    
    def __init__(self):
        self.age_predictor = AgePredictor()
        self.effects = {
            "age_prediction": self._age_prediction_effect,
        }

    def _age_prediction_effect(self, image: np.ndarray):
        return self.age_predictor.process_image(image)

    def process_frame(self, image: np.ndarray, effect: str = "age_prediction"):
        if image is None:
            return np.zeros((480, 640, 3), dtype=np.uint8), None
            
        effect_func = self.effects.get(effect, self.effects["age_prediction"])
        return effect_func(image)
