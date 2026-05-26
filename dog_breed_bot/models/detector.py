import cv2
import torch
from ultralytics import YOLO
from config import YOLO_MODEL_PATH
import os

class DogDetector:
    def __init__(self):
        print(f"🔧 Инициализация детектора YOLO...")
        self.model = YOLO(YOLO_MODEL_PATH)
        print(f"✅ Детектор загружен")
    
    def detect_dogs(self, image_path: str):
        """
        Детектирует всех собак на изображении.
        Возвращает список словарей: {'bbox': (x1, y1, x2, y2), 'confidence': float}
        """
        print(f"🔍 Детекция на изображении: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"❌ Файл не найден: {image_path}")
            return []
        
        results = self.model(image_path)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    # Класс 16 в COCO - собака
                    if cls == 16:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        detections.append({'bbox': (x1, y1, x2, y2), 'confidence': conf})
                        print(f"   🐕 Найдена собака: уверенность {conf:.2f}, bbox: ({x1}, {y1}, {x2}, {y2})")
        
        print(f"   Всего найдено собак: {len(detections)}")
        return detections
    
    def draw_bbox(self, image_path: str, bbox, output_path: str):
        """Рисует прямоугольник на изображении и сохраняет результат."""
        img = cv2.imread(image_path)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.imwrite(output_path, img)
        print(f"   📸 Изображение с рамкой сохранено: {output_path}")
    
    def crop_dog(self, image_path: str, bbox, output_path: str):
        """Обрезает изображение по bounding box и сохраняет."""
        img = cv2.imread(image_path)
        x1, y1, x2, y2 = bbox
        # Проверяем границы
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(output_path, crop)
        print(f"   ✂️ Обрезанное изображение сохранено: {output_path}, размер: {crop.shape}")