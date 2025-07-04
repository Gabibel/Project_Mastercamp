# app/__init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config import Config

db = SQLAlchemy()

def create_app():
    app = Flask(__name__,
                template_folder='templates')
    app.config.from_object(Config)
    db.init_app(app)

    from app.routes.main      import main_bp
    from app.routes.dashboard import dashboard_bp
    from app.routes.rules     import rules_bp
    from app.routes.api       import api_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(rules_bp)
    app.register_blueprint(api_bp)

    with app.app_context():
        db.create_all()

    return app
