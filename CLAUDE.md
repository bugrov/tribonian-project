# foldersum

CLI-программа на Python, которая делает общее саммари по содержимому папки с документами
разных форматов (PDF, JPG/PNG/WebP, позже DOCX/XLSX/PPTX) и, по желанию, саммари по каждому файлу.
Файлы могут быть на русском или английском: язык документа определяется автоматически.
Язык саммари по умолчанию русский (`--lang en` для английского).

## Статус

Реализуется по шагам, после каждого шага остановка на проверки и ревью пользователем.
Шаг 0 (каркас) выполнен. Подробный план: `C:\Users\Sanya\.claude\plans\sequential-sleeping-spring.md`.

## Архитектура

```
src/foldersum/
  cli.py            # argparse: путь|ссылка Drive, --per-file, --backend, --lang, --format, --output, --model
  fetch.py          # gdown.download_folder для ссылок Drive; локальный путь используется как есть
  scan.py           # обход папки, определение типа файла
  langdetect_utils.py  # определение языка документа (ru/en)
  extractors/
    base.py         # Protocol Extractor -> ExtractedDoc(path, kind, text, lang, meta)
    pdf.py          # pypdf; если текстового слоя нет -> рендер страниц (pypdfium2) + OCR
    image.py        # Pillow (webp -> png) + Tesseract (rus+eng)
    text.py         # txt/md/csv
    office.py       # docx/xlsx/pptx (позже)
  summarizers/
    base.py         # Protocol Summarizer: summarize(doc), combine(summaries)
    ollama.py       # локальная LLM (по умолчанию)
    claude.py       # Anthropic API (переключатель --backend claude)
  pipeline.py       # map (по файлам, длинное режется на чанки) -> reduce (общее саммари)
  render.py         # вывод markdown/json
tests/              # pytest, моки бэкендов (без сети и без LLM)
```

Поток: fetch -> scan -> extract -> summarize каждого файла (map) -> общее саммари (reduce) -> вывод.
Ошибка на одном файле не валит прогон: файл помечается как failed.

## Как вызывать

```bash
python -m foldersum <путь|ссылка Drive> [--per-file] [--backend ollama|claude] \
    [--lang ru|en] [--format md|json] [--output FILE] [--model NAME]
```

Бэкенд по умолчанию: `ollama` (локально, модель `qwen2.5:7b`; для 4 ГБ VRAM можно `--model qwen2.5:3b`).

## Окружение

- Python 3.11+ (на машине разработки используется 3.14 в `.venv`).
- Установка: `py -3.14 -m venv .venv && .venv\Scripts\activate && pip install -e ".[dev]"`.
- Tesseract OCR с языками `rus` и `eng` (Windows: установщик UB Mannheim) должен быть в PATH.
- Ollama (https://ollama.com) и модель: `ollama pull qwen2.5:7b`. Адрес задаётся `OLLAMA_HOST`.
- Для `--backend claude`: переменная `ANTHROPIC_API_KEY`. Ключ не хранить в коде и git.

## Git

Remote: `git@github.com:bugrov/tribonian-project.git`, ветка `main`.
Коммит и push только после явного OK пользователя.

# Правила работы
- Перед написанием или правкой Python-кода загрузи скилл python-quality.
- После завершения фичи запусти code-simplifier, затем сделай ревью (code-review).
- Перед коммитом прогони ruff, mypy и pytest.
- Для работы с PDF/DOCX/XLSX/PPTX используй соответствующие скиллы.
