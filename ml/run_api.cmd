@echo off
rem Launch the sentiment API from the ml/ directory (checkpoint path is cwd-relative)
cd /d "%~dp0"
python -m uvicorn src.serve.api:app --port 8000 --ws none
