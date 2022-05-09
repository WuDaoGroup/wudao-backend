from pydantic import BaseModel


class User(BaseModel):
    id: int
    username: str
    password: str
    usertype: int

    class Config:
        orm_mode = True


class UserCreate(BaseModel):
    username: str
    password: str
    usertype: int
