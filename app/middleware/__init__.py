from fastapi.middleware.cors import CORSMiddleware

origins = [
    "*"
]


def init_middleware(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )