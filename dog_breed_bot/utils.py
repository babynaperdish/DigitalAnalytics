import os
import shutil
from pathlib import Path
from telegram import Update
from config import TEMP_DIR, ALLOWED_EXTENSIONS, ALLOWED_MIME_TYPES, MAX_FILE_SIZE_MB


async def is_valid_image(update: Update) -> bool:
    """Проверка, что сообщение содержит изображение допустимого формата и размера."""
    if not update.message.photo and not update.message.document:
        return False

    # Если отправлено как фото (сжатое)
    if update.message.photo:
        # Telegram photo всегда JPEG, размер ~ (проверка размера не точная)
        return True

    # Если отправлено как документ
    document = update.message.document
    if document:
        ext = Path(document.file_name).suffix.lower()
        mime = document.mime_type
        size = document.file_size / (1024 * 1024)

        if ext not in ALLOWED_EXTENSIONS or mime not in ALLOWED_MIME_TYPES:
            return False
        if size > MAX_FILE_SIZE_MB:
            return False
        return True
    return False


async def download_photo(update: Update) -> str:
    """Скачивает фото во временную папку и возвращает путь к файлу."""
    file = None
    if update.message.photo:
        # Берём самое большое фото
        photo = update.message.photo[-1]
        file = await photo.get_file()
    elif update.message.document:
        file = await update.message.document.get_file()

    if not file:
        raise ValueError("No image file found")

    ext = ".jpg" if update.message.photo else Path(update.message.document.file_name).suffix
    file_path = os.path.join(TEMP_DIR, f"user_{update.effective_user.id}_{file.file_unique_id}{ext}")
    await file.download_to_drive(file_path)
    return file_path


def cleanup_temp_files():
    """Удаляет все временные файлы в папке TEMP_DIR."""
    for filename in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
