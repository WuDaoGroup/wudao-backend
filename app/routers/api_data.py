import json
import pathlib
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from fastapi import APIRouter, Form

from app import schemas

router = APIRouter(prefix="/data")

# 返回数据信息(所有条目)
@router.get("/analysis/content")
async def return_data_file_info(username: str):
    df = pd.read_csv(f"./static/data/{username}/data.csv")
    res = df.to_json(orient="records")
    parsed = json.loads(res)
    for idx, p in enumerate(parsed):
        p["id"] = idx
    # print(parsed)
    header = []
    for idx, e in enumerate(df.columns):
        h = {}
        h["key"] = e
        h["value"] = e
        header.append(h)
    response = {
        "header": header,
        "content": parsed,
    }
    return response


# 将原data根据传入的feature info list，筛选出target/feature
@router.post("/features/info")
async def process_selected_features(info: List[schemas.FeatureInfo], username: str):
    all_features = []
    df = pd.read_csv(f"./static/data/{username}/data.csv")
    features = []
    for i in info:
        if i.type == "target":
            all_features.append(i)
            features.append(i.value)
            break
    for i in info:
        if i.type == "feature":
            all_features.append(i)
            features.append(i.value)
    df = df[features]
    df.to_csv(f"./static/data/{username}/data_target_confirmed.csv", index=False)
    response = {
        "target": features[0],
        "features": features[1:],
        "all_features": all_features,
    }
    return response


@router.post("/statistics/info")
def show_data_statistics_info(username: str = Form(...), step: str = Form(...)):
    df = pd.read_csv(f"./static/data/{username}/{step}.csv")
    len_df = len(df.index)
    header = [
        {"key": "name", "value": "name"},
        {"key": "count", "value": "count"},
        {"key": "missing", "value": "missing"},
        {"key": "min", "value": "min"},
        {"key": "max", "value": "max"},
        {"key": "mean", "value": "mean"},
        {"key": "std", "value": "std"},
        {"key": "median", "value": "median"},
    ]
    statistic_info = []
    for idx, e in enumerate(df.columns):
        h = {}
        h["id"] = idx
        h["name"] = e
        h["count"] = int(df[e].count())
        h["missing"] = str(round(float((100 - df[e].count() * 100 / len_df)), 2)) + "%"
        h["mean"] = round(float(df[e].mean()), 2)
        h["max"] = round(float(df[e].max()), 2)
        h["min"] = round(float(df[e].min()), 2)
        h["median"] = round(float(df[e].median()), 2)
        h["std"] = round(float(df[e].std()), 2)
        statistic_info.append(h)

    response = {
        "header": header,
        "content": statistic_info,
    }
    return response


@router.get("/zscore")
def zscore_data(username: str):
    df = pd.read_csv(f"./static/data/{username}/data_target_confirmed.csv")
    # 第一列是预测目标y，跳过，不能被标准化
    df_features = df.iloc[:, 1:]
    df_features = df_features.apply(lambda x: (x - x.mean()) / (x.std() + 1e-12))
    df = pd.concat([df.iloc[:, 0], df_features], axis=1)
    df.to_csv(f"./static/data/{username}/data_zscore.csv", index=False)
    res = {"message": "success"}
    return res


@router.get("/fill")
def fill_data(username: str, fill_type: str):
    df = pd.read_csv(f"./static/data/{username}/data_zscore.csv")
    if fill_type == "均值填充":
        # 因为已经zscore好了，所以只需补0即可，0即为均值
        df.fillna(value=0, inplace=True)
    elif fill_type == "中位数填充":
        # 先计算出原始feature的中位数
        info = []
        for _, e in enumerate(df.columns):
            h = {}
            h["name"] = e
            h["median"] = float(df[e].median())
            info.append(h)
        print(info)
        for idx, e in enumerate(df.columns):
            df[e].fillna(value=info[idx]["median"], inplace=True)
    df.to_csv(f"./static/data/{username}/data_zscore_fill.csv", index=False)
    res = {"message": "success"}
    return res


@router.get("/filter")
def filter_data(username: str, bar: float):
    df = pd.read_csv(f"./static/data/{username}/data_zscore_fill.csv")
    # 除第一列外，如果该行存在大于bar的值，则删除该行
    for f in df.iloc[:, 1:].columns:
        df = df[abs(df[f]) < bar]
    df.to_csv(f"./static/data/{username}/data_zscore_fill_filter.csv", index=False)
    res = {"message": "success"}
    return res


@router.post("/histogram")
def generate_histogram(username: str = Form(...), step: str = Form(...)):
    df = pd.read_csv(f"./static/data/{username}/{step}.csv")
    for _, f in enumerate(df.columns):
        plt.figure(figsize=(10, 6))
        plt.title(f, fontsize=18)
        # plt.hist(df[f],bins=20,edgecolor='k',alpha=0.5)
        # plt.xticks(rotation=90)
        sns.histplot(data=df[f], color="dodgerblue")
        pathlib.Path(f"./static/data/{username}/images/{step}").mkdir(
            parents=True, exist_ok=True
        )
        plt.savefig(f"./static/data/{username}/images/{step}/histogram_{f}.png")
        plt.clf()
    res = {"message": "success"}
    return res
