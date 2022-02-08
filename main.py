import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

import crud, models, schemas
from database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.middleware("http")
async def db_session_middleware(request: Request, call_next):
    response = Response("Internal server error", status_code=500)
    try:
        request.state.db = SessionLocal()
        response = await call_next(request)
    finally:
        request.state.db.close()
    return response

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
async def root():
    return {"msg": "Hello, World!"}

@app.get("/test")
async def test(token: str = Depends(oauth2_scheme)):
    return {"token": token}

@app.get("/users", response_model=list[schemas.User])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users

@app.get("/users/{user_id}", response_model=schemas.User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@app.post("/users/login", response_model=schemas.User)
def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    db_user = crud.get_user_by_username(db, username=username)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if db_user.password != password:
        raise HTTPException(status_code=250, detail="Password not correct")
    return db_user

@app.post("/users/register", response_model=schemas.UserCreate)
def register(username: str = Form(...), password1: str = Form(...),password2: str = Form(...), db: Session = Depends(get_db)):
    db_user = crud.get_user_by_username(db, username=username)
    if db_user is not None:
        raise HTTPException(status_code=404, detail="User already registered")
    if password1 != password2:
        raise HTTPException(status_code=250, detail="密码不同")
    new_user = schemas.UserCreate(username=username, password=password1, usertype=0)
    crud.create_user(db,new_user)
    return new_user


if __name__ == "__main__":
    uvicorn.run(app="main:app", host="127.0.0.1", port=8123, reload=True)
