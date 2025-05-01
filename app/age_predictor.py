
import cv2
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import tensorflow as tf
from tensorflow.keras.models import load_model

AGE_RANGES = [
    (0, 2),    
    (3, 9),    
    (10, 19),
    (20, 29),  
    (30, 39),  
    (40, 49),  
    (50, 59),  
    (60, 100)
]
class AgePredictor:
    def __init__(self):
        self.face_model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt") # https://huggingface.co/AdamCodd/YOLOv11n-face-detection
        self._initialize_model()
        self.face_class_id = 0

    def _initialize_model(self):
        self.face_model = YOLO(self.face_model_path)
        self.age_model = load_model('models/age_cnn_model_final.h5')

    def preditct_age(self, image: np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (128, 128))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        pred_probs = self.age_model.predict(img_batch, verbose=0)[0]
        pred_class = np.argmax(pred_probs)
        
        age_range = AGE_RANGES[pred_class]
        confidence = float(pred_probs[pred_class])

        return age_range, confidence

    def process_image(self, image: np.ndarray):
        image = np.array(image, dtype=np.uint8).copy()
        results = self.face_model(image, conf=0.5)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls[0]) == self.face_class_id:
                    age_range, confidence_score = self.preditct_age(image)
                    self._draw_detection_box(image, box, age_range, confidence_score)
        
        return image
        
    def _draw_detection_box(self, image: np.ndarray, box, age_range, confidence_score):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        width = x2 - x1
        height = y2 - y1
        max_side = max(width, height)
        
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        
        x1 = x_center - max_side // 2
        x2 = x_center + max_side // 2
        y1 = y_center - max_side // 2
        y2 = y_center + max_side // 2

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'age range: {age_range}', (x1, y1 - 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(image, f'confidence score: {confidence_score:.2f}', (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    