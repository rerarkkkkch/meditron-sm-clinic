@echo off
chcp 65001 >nul
rem запуск демо на Windows: двойной клик по файлу
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Создаю окружение .venv ...
  py -3 -m venv .venv 2>nul || python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
  echo Не найден Python 3.10+. Установите его с python.org и отметьте "Add python.exe to PATH".
  pause
  exit /b 1
)
echo Устанавливаю зависимости (первый раз 1-3 минуты) ...
".venv\Scripts\python.exe" -m pip install -q --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
  echo Не удалось установить зависимости, смотрите сообщение выше.
  pause
  exit /b 1
)
set CM_ENV=demo
echo.
echo Сервис запускается: http://localhost:8000  (остановить: Ctrl+C)
start "" cmd /c "timeout /t 5 >nul & start http://localhost:8000"
".venv\Scripts\python.exe" -m uvicorn app.main:app --port 8000
pause
