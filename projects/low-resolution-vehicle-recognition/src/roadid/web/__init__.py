"""RoadID Flask application and worker lifecycle."""

from roadid.web.app import create_app

__all__ = ["create_app"]


def main() -> None:
    from roadid.settings import load_settings

    settings = load_settings()
    app = create_app(settings=settings)
    try:
        app.run(
            host=settings.web.host,
            port=settings.web.port,
            debug=settings.web.debug,
            threaded=True,
            use_reloader=False,
        )
    finally:
        app.extensions["roadid"]["shutdown"]()
