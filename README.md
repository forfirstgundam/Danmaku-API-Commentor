# Danmaku AI Overlay

Python foundation for an AI-generated danmaku overlay project.

## Modules

- `capture`: screen capture
- `api`: Gemini API client and prompt builder
- `overlay`: transparent click-through danmaku overlay
- `ui`: minimal settings window
- `main.py`: connects all modules

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app

Dummy API mode is enabled by default, so the app can run without an API key.

```bash
set PYTHONPATH=src
python -m danmaku.main
```

## Run module demos

```bash
set PYTHONPATH=src
python -m danmaku.capture.capture_service
python -m danmaku.api.llm_client
python -m danmaku.overlay.overlay_window
python -m danmaku.ui.settings_window
```

## Using Gemini

Set environment variables:

```bash
set GEMINI_API_KEY=your_key_here
set DANMAKU_USE_DUMMY_API=false
set MODEL_NAME=gemini-2.5-flash-lite
set CAPTURE_MODE=region
set CAPTURE_REGION=320,180,1280,720
set PYTHONPATH=src
python -m danmaku.main
```

## Build EXE with PyInstaller

Install PyInstaller:

```bash
pip install pyinstaller
```

Recommended first build:

```bash
pyinstaller --onedir --name DanmakuAI --paths src --add-data "prompts/system_prompt.txt;prompts" src/danmaku/main.py
```

One-file build after debugging:

```bash
pyinstaller --onefile --noconsole --name DanmakuAI --paths src --add-data "prompts/system_prompt.txt;prompts" src/danmaku/main.py
```

During debugging, omit `--noconsole` so errors are visible.
