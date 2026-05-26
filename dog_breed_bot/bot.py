import os
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler, ContextTypes

from config import BOT_TOKEN, TEMP_DIR
from utils import is_valid_image, download_photo, cleanup_temp_files
from models.detector import DogDetector
from models.classifier import DogBreedClassifier
from breeds import load_breed_names

# Загружаем список пород
BREED_NAMES = load_breed_names("breed_names.txt")
print(f"📋 В BREED_NAMES загружено: {len(BREED_NAMES)} пород")

# Инициализация моделей
detector = DogDetector()
classifier = DogBreedClassifier("models/best_model.pth", BREED_NAMES)

# Клавиатура главного меню
main_keyboard = ReplyKeyboardMarkup(
    [["🐕 Загрузить фотографию", "ℹ️ О боте"]],
    resize_keyboard=True
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start."""
    await update.message.reply_text(
        "Привет! Я бот для распознавания пород собак.\n"
        "Отправьте фото собаки, и я определю её породу.\n\n"
        "Используйте кнопки меню для навигации.",
        reply_markup=main_keyboard
    )


async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик кнопки 'О боте'."""
    text = (
        "🐕 *Система распознавания пород собак*\n\n"
        "Модель детекции: YOLOv8\n"
        "Модель классификации: EfficientNet-B4 (обучена на Stanford Dogs)\n"
        "Точность на тестовой выборке: ~85-90%\n\n"
        "Поддерживаемые форматы: JPEG, PNG, до 15 МБ\n"
        "Результат: топ-3 наиболее вероятные породы с процентами уверенности."
    )
    await update.message.reply_text(text, parse_mode="Markdown", reply_markup=main_keyboard)


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Основная обработка изображения."""
    if not await is_valid_image(update):
        await update.message.reply_text(
            "❌ Неверный формат файла.\n"
            "Пожалуйста, отправьте изображение в формате JPEG или PNG размером до 15 МБ."
        )
        return

    processing_msg = await update.message.reply_text("⏳ Анализирую изображение, подождите...")

    try:
        original_path = await download_photo(update)
    except Exception:
        await processing_msg.edit_text("❌ Не удалось загрузить фото. Попробуйте ещё раз.")
        return

    try:
        detections = detector.detect_dogs(original_path)

        if not detections:
            await processing_msg.edit_text(
                "🐕‍🦺 На фото не обнаружено собаки. Пожалуйста, загрузите другое изображение."
            )
            cleanup_temp_files()
            return

        context.user_data['original_path'] = original_path
        context.user_data['detections'] = detections

        if len(detections) == 1:
            await process_single_dog(update, context, processing_msg, 0)
        else:
            keyboard = []
            for i, dog in enumerate(detections):
                keyboard.append([InlineKeyboardButton(
                    f"Собака {i + 1} (уверенность {dog['confidence']:.0%})",
                    callback_data=str(i)
                )])
            keyboard.append([InlineKeyboardButton("Всех собак", callback_data="all")])
            reply_markup = InlineKeyboardMarkup(keyboard)
            await processing_msg.edit_text(
                f"🔍 На фото найдено {len(detections)} собак. Для какой определить породу?",
                reply_markup=reply_markup
            )
    except Exception as e:
        await processing_msg.edit_text("⚠️ Сервис временно недоступен, попробуйте позже.")
        print(f"Error: {e}")
        cleanup_temp_files()


async def process_single_dog(update: Update, context: ContextTypes.DEFAULT_TYPE, msg, dog_index: int):
    """Обрабатывает одну собаку по индексу."""
    original_path = context.user_data.get('original_path')
    detections = context.user_data.get('detections')
    
    if not original_path or not detections:
        # Пробуем отправить ответ через разные каналы
        try:
            if msg:
                await msg.edit_text("❌ Ошибка: данные утеряны. Отправьте фото заново.")
            elif update.callback_query:
                await update.callback_query.message.edit_text("❌ Ошибка: данные утеряны. Отправьте фото заново.")
            elif update.effective_chat:
                await update.effective_chat.send_message("❌ Ошибка: данные утеряны. Отправьте фото заново.")
        except Exception:
            pass
        return
    
    dog = detections[dog_index]
    crop_path = os.path.join(TEMP_DIR, f"crop_{update.effective_user.id}_{dog_index}.jpg")
    
    try:
        detector.crop_dog(original_path, dog['bbox'], crop_path)
        top3 = classifier.predict(crop_path, top_k=3)
    except Exception as e:
        print(f"Ошибка: {e}")
        try:
            if msg:
                await msg.edit_text("⚠️ Ошибка классификации. Попробуйте другое фото.")
            elif update.callback_query:
                await update.callback_query.message.edit_text("⚠️ Ошибка классификации. Попробуйте другое фото.")
        except Exception:
            pass
        cleanup_temp_files()
        return
    
    output_path = os.path.join(TEMP_DIR, f"result_{update.effective_user.id}.jpg")
    detector.draw_bbox(original_path, dog['bbox'], output_path)
    
    # Формируем ответ
    result_text = f"🐕 РЕЗУЛЬТАТ РАСПОЗНАВАНИЯ:\n\n"
    for i, (breed, prob) in enumerate(top3):
        result_text += f"{i+1}. {breed}: {prob:.1f}%\n"
    
    # Определяем, откуда пришёл вызов (из handle_image или из button_callback)
    if update.callback_query:
        # Это вызов из кнопки - отвечаем через callback_query
        chat_id = update.effective_chat.id
        try:
            # Удаляем сообщение с кнопками
            await update.callback_query.message.delete()
        except Exception:
            pass
        
        # Отправляем результат в чат
        try:
            with open(output_path, 'rb') as photo:
                await context.bot.send_photo(
                    chat_id=chat_id,
                    photo=photo,
                    caption=result_text,
                    reply_markup=main_keyboard
                )
        except Exception as e:
            print(f"Ошибка при отправке фото через bot: {e}")
            await context.bot.send_message(chat_id=chat_id, text=result_text, reply_markup=main_keyboard)
    else:
        # Это вызов из handle_image - есть update.message
        try:
            if msg:
                try:
                    await msg.delete()
                except Exception:
                    pass
            
            with open(output_path, 'rb') as photo:
                await update.message.reply_photo(
                    photo=photo,
                    caption=result_text,
                    reply_markup=main_keyboard
                )
        except Exception as e:
            print(f"Ошибка при отправке фото через message: {e}")
            try:
                await update.message.reply_text(result_text, reply_markup=main_keyboard)
            except Exception:
                pass
    
    cleanup_temp_files()

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка нажатия инлайн-кнопок (выбор собаки)."""
    query = update.callback_query
    await query.answer()

    data = query.data
    original_path = context.user_data.get('original_path')
    detections = context.user_data.get('detections')

    if not original_path or not detections:
        await query.message.edit_text("❌ Ошибка: данные утеряны. Отправьте фото заново.")
        return

    if data == "all":
        await query.message.edit_text("Обрабатываю всех собак, это может занять некоторое время...")
        for i, dog in enumerate(detections):
            crop_path = os.path.join(TEMP_DIR, f"crop_{update.effective_user.id}_{i}.jpg")
            detector.crop_dog(original_path, dog['bbox'], crop_path)
            top3 = classifier.predict(crop_path, top_k=3)
            result_text = f"🐕 СОБАКА {i + 1} (уверенность {dog['confidence']:.0%}):\n\n"
            for breed, prob in top3:
                result_text += f"• {breed}: {prob:.1f}%\n"
            await update.effective_chat.send_message(result_text)
        cleanup_temp_files()
        await query.message.delete()
    else:
        dog_idx = int(data)
        # Передаём None вместо msg, так как это callback
        await process_single_dog(update, context, None, dog_idx)
        
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Глобальный обработчик ошибок."""
    print(f"Exception: {context.error}")
    if update and update.effective_message:
        try:
            await update.effective_message.reply_text("⚠️ Произошла ошибка. Попробуйте позже.")
        except Exception:
            pass


def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Regex("^🐕 Загрузить фотографию$"),
                                   lambda u, c: u.message.reply_text("📸 Пожалуйста, отправьте фото собаки в хорошем качестве.")))
    app.add_handler(MessageHandler(filters.Regex("^ℹ️ О боте$"), about))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_image))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_error_handler(error_handler)

    print("Бот запущен...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()