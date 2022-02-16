import uvicorn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from fastapi import Depends, FastAPI, HTTPException, Request, Response, Form, File, UploadFile
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pandas.api.types import CategoricalDtype
from io import StringIO

import crud, models, schemas
import pandas as pd
import numpy as np
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
        raise HTTPException(status_code=250, detail="User not found")
    return db_user

@app.post("/users/login", response_model=schemas.User)
def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    db_user = crud.get_user_by_username(db, username=username)
    if db_user is None:
        raise HTTPException(status_code=250, detail="User not found")
    if db_user.password != password:
        raise HTTPException(status_code=251, detail="Password not correct")
    return db_user

@app.post("/users/register", response_model=schemas.UserCreate)
def register(username: str = Form(...), password1: str = Form(...),password2: str = Form(...), db: Session = Depends(get_db)):
    db_user = crud.get_user_by_username(db, username=username)
    if db_user is not None:
        raise HTTPException(status_code=250, detail="User already registered")
    if password1 != password2:
        raise HTTPException(status_code=251, detail="Passwords are not the same")
    new_user = schemas.UserCreate(username=username, password=password1, usertype=0)
    crud.create_user(db,new_user)
    return new_user

@app.post("/predict/ols")
def ordinary_least_squares(filename: str = Form(...)):
    res = {}
    data = pd.read_csv('./data/'+ filename)
    reg = linear_model.LinearRegression()
    X = data.iloc[:, 1:]
    y = data.iloc[:, :1]
    reg.fit( X, y )
    reg_list1 = reg.coef_.tolist()
    reg_list2 = reg.intercept_.tolist()
    res["result_coef"] = reg_list1[0]
    res["result_intercept"] = reg_list2[0]
    return res

@app.post("/predict/bdtr")
def boosted_decision_tree_regression( filename: str = Form(...), id: str = Form(...)):
    the_addr = 'C:\\Users\\DELL\\Desktop\\wudao-backend\\pictures\\' + filename + id + '_handle_result.png'
    res = {}
    data = pd.read_csv('./data/'+ filename) 
    rng = np.random.RandomState(1)
    X = data.iloc[:, 1:2]
    y = data.iloc[:, 0:1]
    regr_1 = DecisionTreeRegressor(max_depth=4)
    regr_2 = AdaBoostRegressor(
        DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=rng
    )
    regr_1.fit(X, y)
    regr_2.fit(X, y)
    y_1 = regr_1.predict(X)
    y_2 = regr_2.predict(X)
    plt.figure()
    plt.scatter(X, y, c="k", label="training samples")
    plt.plot(X, y_1, c="g", label="n_estimators=1", linewidth=2)
    plt.plot(X, y_2, c="r", label="n_estimators=300", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Boosted Decision Tree Regression")
    plt.legend()
    plt.savefig(the_addr)
    
    res['pic_addr'] = the_addr
    return res

@app.get("/image/{filename}")
def get_image(filename: str):
    img_addr = 'images/test_image.png'
    return FileResponse(path=img_addr, filename='hhh_img.jpeg', media_type='image/png')

@app.get("/files/upload")
async def create_upload_file(upload_file: UploadFile = File(...)):
    file_location = f"./data/{upload_file.filename}"
    print('get_file:',file_location)
    with open(file_location, "wb+") as file_object:
        file_object.write(upload_file.file.read())
    return {"info": f"file '{upload_file.filename}' saved at '{file_location}'"}

if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=8123, reload=True)
