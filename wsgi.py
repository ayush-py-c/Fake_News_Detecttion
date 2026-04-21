import importlib.util
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_PATH = os.path.join(BASE_DIR, "Model deployment using Flask", "app.py")

spec = importlib.util.spec_from_file_location("fake_news_flask_app", APP_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load Flask app from {APP_PATH}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

app = module.app