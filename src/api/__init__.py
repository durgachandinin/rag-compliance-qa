__all__ = ["app"]

def get_app():
    from src.api.main import app
    return app
