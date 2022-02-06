from sqlalchemy.orm import Session

import models, schemas

def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()

def create_user(db: Session, user: schemas.User):
    username = user.username
    password = user.password
    usertype = user.usertype
    db_user = models.User(username=username, password=password, usertype=usertype)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user