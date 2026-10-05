# загрузка настроек из YAML с автоматической перезагрузкой при изменении файла
import os                     # работа с путями и временем изменения файлов
import threading              # блокировка, чтобы два запроса не перечитывали файл одновременно
import yaml                   # чтение YAML-файлов

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # корень проекта (папка над app/)
CONFIG_DIR = os.environ.get("CM_CONFIG_DIR", os.path.join(BASE_DIR, "config"))  # папку конфигов можно переопределить переменной окружения

_cache = {}                   # кэш: имя файла -> (время изменения, содержимое)
_lock = threading.Lock()      # замок для потокобезопасности


def load(name: str) -> dict:
    # вернуть содержимое config/<name>.yaml; перечитать, если файл изменился
    path = os.path.join(CONFIG_DIR, f"{name}.yaml")              # полный путь к файлу
    mtime = os.path.getmtime(path)                               # время последнего изменения файла
    with _lock:                                                  # входим в критическую секцию
        cached = _cache.get(name)                                # смотрим, что уже лежит в кэше
        if cached and cached[0] == mtime:                        # файл не менялся с прошлого чтения
            return cached[1]                                     # отдаём закэшированную версию
        try:
            with open(path, encoding="utf-8") as f:              # открываем файл в UTF-8 (русский текст)
                data = yaml.safe_load(f) or {}                   # безопасно парсим YAML (без исполнения кода)
            _validate(name, data)                                # пустой/битый словарь = тихий ноль находок
        except Exception as e:
            if cached:                                           # при «горячей» правке с ошибкой держим последнюю рабочую версию
                print(f"[config] {name}.yaml не принят ({e}); работаю на предыдущей версии", flush=True)
                _cache[name] = (mtime, cached[1])
                return cached[1]
            raise                                                # при старте — падаем громко
        _cache[name] = (mtime, data)                             # кладём в кэш вместе с временем изменения
        return data                                              # возвращаем свежие настройки


def _validate(name: str, data: dict):
    # минимальная проверка структуры. Ошибка — исключение (сервис не стартует / правка не применяется)
    if name == "findings":
        fs = data.get("findings") or {}
        if not fs:
            raise ValueError("в findings.yaml нет ни одной находки")
        locs = set(data.get("locations") or {})
        for fid, f in fs.items():
            if not f.get("route") or not (f.get("terms") or f.get("what") or f.get("ml_only")):
                raise ValueError(f"находка {fid}: нужны route и terms/what (или ml_only: true)")
            bad = set(f.get("where") or []) - locs
            if bad:
                raise ValueError(f"находка {fid}: неизвестные органы {bad}")
    if name == "routes":
        if not data.get("routes") or not data.get("templates") or "behavior" not in data:
            raise ValueError("в routes.yaml нет routes/templates/behavior")


def findings() -> dict:
    # словарь находок (config/findings.yaml)
    return load("findings")                                      # короткий доступ к словарю


def routes() -> dict:
    # матрица маршрутизации и тексты (config/routes.yaml)
    return load("routes")                                        # короткий доступ к маршрутам


def economics() -> dict:
    # допущения для экономического расчёта (config/economics.yaml)
    return load("economics")                                     # короткий доступ к экономике
