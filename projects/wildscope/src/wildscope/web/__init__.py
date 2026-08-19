"""WildScope web application."""

from wildscope.web.app import create_app

__all__ = ["create_app"]


def main() -> None:
    import os

    app = create_app()
    app.run(
        host=os.getenv("WILDSCOPE_HOST", "127.0.0.1"),
        port=int(os.getenv("WILDSCOPE_PORT", "5000")),
        debug=False,
        threaded=True,
        use_reloader=False,
    )
