@echo off
cd /d D:\Code\Multi-Agent-Medical-Assistant
call .venv\Scripts\activate.bat > startup.log 2>&1
.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload >> startup.log 2>&1
