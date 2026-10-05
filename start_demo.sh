#!/usr/bin/env bash
# запуск демо на macOS / Linux: ./start_demo.sh
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Создаю окружение .venv ..."
  python3 -m venv .venv
fi
echo "Устанавливаю зависимости (первый раз 1-3 минуты) ..."
.venv/bin/python -m pip install -q --disable-pip-version-check -r requirements.txt
echo "Сервис запускается: http://localhost:8000  (остановить: Ctrl+C)"
CM_ENV=demo exec .venv/bin/python -m uvicorn app.main:app --port 8000
