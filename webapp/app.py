from __future__ import annotations

from pathlib import Path

from flask import Flask, render_template

import imagemessrs.effects  # noqa: F401  (populates the effect registry on import)

from .cleanup import start_cleanup_thread

BASE_DIR = Path(__file__).resolve().parent.parent


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024  # 256MB uploads (video)

    start_cleanup_thread()

    from .routes.images import images_bp
    from .routes.video import video_bp

    app.register_blueprint(images_bp)
    app.register_blueprint(video_bp)

    @app.route("/")
    def root():
        return render_template("index.html")

    @app.errorhandler(404)
    def not_found(_error):
        return render_template("not_found.html"), 404

    return app
