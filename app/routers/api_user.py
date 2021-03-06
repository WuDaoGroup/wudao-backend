import hashlib
from typing import List

from fastapi import APIRouter, Depends, Form, HTTPException
from sqlalchemy.orm import Session

from app import crud, schemas
from app.database import SessionLocal

router = APIRouter(prefix="/users")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/", response_model=List[schemas.User])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users


@router.get("/{user_id}", response_model=schemas.User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=250, detail="User not found")
    return db_user


@router.post("/login", response_model=schemas.User)
def login(
    username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)
):
    db_user = crud.get_user_by_username(db, username=username)
    encode_password = hashlib.sha256(password.encode("utf-8")).hexdigest()
    if db_user is None:
        raise HTTPException(status_code=250, detail="User not found")
    if db_user.password != encode_password:
        raise HTTPException(status_code=251, detail="Password not correct")
    return db_user


@router.post("/register", response_model=schemas.UserCreate)
def register(
    username: str = Form(...),
    password1: str = Form(...),
    password2: str = Form(...),
    db: Session = Depends(get_db),
):
    db_user = crud.get_user_by_username(db, username=username)
    if db_user is not None:
        raise HTTPException(status_code=250, detail="User already registered")
    if password1 != password2:
        raise HTTPException(status_code=251, detail="Passwords are not the same")
    encode_password = hashlib.sha256(password1.encode("utf-8")).hexdigest()
    new_user = schemas.UserCreate(
        username=username, password=encode_password, usertype=0
    )
    crud.create_user(db, new_user)
    return new_user
