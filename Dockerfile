# образ сервиса: всё работает внутри контейнера, наружу ничего не ходит
FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Europe/Moscow
WORKDIR /app
# отдельный непривилегированный пользователь
RUN useradd --system --uid 10001 --no-create-home app
COPY requirements.txt .
# в проде ставить из lock-файла: pip install --require-hashes -r requirements.lock
RUN pip install -r requirements.txt
# код и настройки принадлежат root: приложение не может изменить само себя
COPY app ./app
COPY config ./config
COPY static ./static
# второй контур: обученная модель (scripts/train.py). Обучена на обезличенных текстах, проверена на отсутствие ФИО
COPY models ./models
RUN mkdir -p data/db data/inbox/processed data/inbox/failed && chown -R app:app data
USER app
EXPOSE 8000
# проверка живости (curl в slim нет — используем python)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4).status==200 else 1)"
# ровно один воркер — блокировка записи SQLite и фоновые потоки рассчитаны на один процесс
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-server-header"]

# рабочий образ: без синтетики и скриптов, «Демо-данные» в бою не работают
FROM base AS prod

# демо/хакатон: синтетика, скрипты, второй слой
FROM base AS demo
USER root
COPY scripts ./scripts
COPY data/synthetic ./data/synthetic
USER app
ENV CM_ENV=demo
