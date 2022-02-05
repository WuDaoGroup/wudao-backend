import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"msg": "Hello, World!"}


if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=8123, reload=True)
