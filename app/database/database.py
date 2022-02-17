from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "postgresql://pujsuqaoqnrzyz:8e977decc5189564c1b10562041b0de3062e91aee8ca316d12c800809b5a6b82@ec2-35-175-68-90.compute-1.amazonaws.com:5432/ddd9r0vs09pdtg"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()