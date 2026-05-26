import os

# Telegram Bot Token
BOT_TOKEN = "8772553584:AAEsIRNoevGBC145_leUp2kHm8Eds0otZQM"  # замените на свой

# Пути к моделям
YOLO_MODEL_PATH = "yolov8n.pt"
CLASSIFIER_MODEL_PATH = "models/best_model.pth"

# Параметры
MAX_FILE_SIZE_MB = 15
ALLOWED_MIME_TYPES = ["image/jpeg", "image/png"]
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# Временная папка
TEMP_DIR = "temp_images"
os.makedirs(TEMP_DIR, exist_ok=True)

# Конфигурация для классификатора
CLASSIFIER_CONFIG = {
    "model_path": "models/best_model.pth",
    "img_size": 224
}