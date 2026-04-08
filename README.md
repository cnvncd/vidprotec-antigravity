# 🛡️ StealthMask — сделай фото и видео невидимыми для ИИ

StealthMask накладывает **невидимый для человека** adversarial-шум на изображения и видео,
полностью ломая OCR, object detection, scene understanding и распознавание речи у любых ИИ-моделей
(GPT-4V, Claude Vision, Tesseract, Whisper, Deepgram и др.).

Ты видишь обычную картинку — ИИ видит хаос.

---

## ⚡ Быстрый старт

### 1. Установи Python 3.11+

Скачай с [python.org](https://www.python.org/downloads/) (убедись что `python` и `pip` в PATH).

### 2. Установи FFmpeg

**Windows:**
```bash
# Через winget
winget install Gyan.FFmpeg

# Или скачай с https://ffmpeg.org/download.html и добавь в PATH
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update && sudo apt install ffmpeg -y
```

Проверь установку: `ffmpeg -version`

### 3. Установи зависимости Python

```bash
cd vidprotec-antigravity
pip install -r requirements.txt
```

### 4. Запуск

```bash
python app.py
```

Открой в браузере: **http://localhost:5000**

---

## 🎯 Возможности

| Функция | Описание |
|---------|----------|
| 🖼️ Adversarial-шум для фото | Ломает OCR, object detection, scene understanding |
| 🎬 Покадровая обработка видео | Каждый кадр получает adversarial perturbation |
| 🔇 Маскировка аудио | Неслышимый шум, который ломает Whisper/Deepgram |
| 📊 Настройка силы | Слайдер 1–10 для контроля интенсивности |
| 📦 Пакетная обработка | Загружай десятки файлов одновременно |
| 🔍 Сравнение До/После | Side-by-side просмотр результатов |
| 📥 Скачивание ZIP | Все обработанные файлы одним архивом |

## 📁 Поддерживаемые форматы

- **Изображения:** JPG, PNG, WEBP
- **Видео:** MP4, MOV, AVI

## 🏗️ Структура проекта

```
vidprotec-antigravity/
├── app.py                  # Flask-сервер + вся логика обработки
├── requirements.txt        # Python-зависимости
├── .env                    # Конфигурация (порт, debug и т.д.)
├── README.md               # Этот файл
├── templates/
│   └── index.html          # Главная страница (SPA)
└── static/
    ├── styles.css           # Кастомные стили
    └── script.js            # Логика фронтенда
```

## ⚙️ Настройка

Параметры в `.env`:

```env
FLASK_PORT=5000
FLASK_DEBUG=false
MAX_CONTENT_MB=500
```

## 📝 Лицензия

MIT — используй как хочешь.
