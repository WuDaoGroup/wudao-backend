from app.database import engine

from .model_user import *


def init_models():
    Base.metadata.create_all(bind=engine)
