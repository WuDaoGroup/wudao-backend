from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.middleware import init_middleware
from app.routers import init_routers

def create_app():
    app = FastAPI(title="悟道-WuDao",
                  description="FastAPI 后端",
                  version="0.0.1"
                  )
    
    app.mount("/static", StaticFiles(directory="static"), name="static")

    init_middleware(app)
    init_routers(app)
    return app
