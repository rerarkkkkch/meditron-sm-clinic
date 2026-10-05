import os, sys, tempfile
os.environ.setdefault("CM_ENV", "demo")                  # модельное время и демо-соль
os.environ["CM_DB"] = tempfile.mktemp(suffix=".db")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
