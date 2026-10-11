@echo off
cd /d "%~dp0\.."
python -u web\app.py --host 0.0.0.0 --port 5000 %*
