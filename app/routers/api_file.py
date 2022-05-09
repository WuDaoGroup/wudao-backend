import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from sklearn.preprocessing import LabelEncoder

plt.rcParams["font.sans-serif"] = ["SimHei"]  ##设置字体为 黑体
plt.rcParams["axes.unicode_minus"] = False  ##显示符号
router = APIRouter(prefix="/files")


# 上传文件，根据文件后缀名的不同，直接将文件转换成csv格式并保存到本地
@router.post("/upload")
async def create_upload_file(username: str, upload_file: UploadFile = File(...)):
    file_type = os.path.splitext(upload_file.filename)[1]
    if file_type not in [".csv", ".xls", ".xlsx"]:
        raise HTTPException(status_code=240, detail="File type not correct")
    file_location = f"./static/data/{upload_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(upload_file.file.read())  # 原始数据
    # 若为Excel文件，则转换成csv文件
    if file_type in [".xls", ".xlsx"]:
        df = pd.read_excel(file_location)
    else:
        df = pd.read_csv(file_location)
    # 数值化
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    le = LabelEncoder()
    for col in non_numeric_columns:
        df[col] = le.fit_transform(df[col])
    pathlib.Path(f"./static/data/{username}").mkdir(parents=True, exist_ok=True)
    df.to_csv(f"./static/data/{username}/data.csv", index=False)
