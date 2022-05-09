import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from autogluon.tabular import TabularPredictor
from catboost import CatBoostRegressor
from fastapi import APIRouter, Form
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    plot_roc_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier

# import lightgbm as lgb
# import autosklearn.regression
# import autosklearn.classification


router = APIRouter(prefix="/models")


@router.post("/regression/train")
def train_regression_model(
    username: str = Form(...), percent: float = Form(...), method: str = Form(...)
):
    # 读数据文件
    df = pd.read_csv(f"./static/data/{username}/data_zscore_fill_filter.csv")

    # 划分训练集和测试集
    x = df.iloc[:, 1:]
    y = df.iloc[:, :1]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=percent, random_state=42
    )

    # 选择模型
    if method == "xgboost":
        model = xgb.XGBRegressor(verbosity=0, n_estimators=100, learning_rate=0.1)
        # n_estimators – Number of gradient boosted trees. Equivalent to number of boosting rounds.
        # 训练模型，对于回归模型使用r2评价指标
        model.fit(x_train, y_train, eval_metric="auc")
        pathlib.Path(f"./static/data/{username}/images/{method}").mkdir(
            parents=True, exist_ok=True
        )
        xgb.plot_importance(model, max_num_features=10, importance_type="gain")
        plt.title("Feature Importance")
        plt.savefig(f"./static/data/{username}/images/{method}/feature_importance.png")
        plt.clf()
        xgb.plot_tree(model)
        plt.gcf().set_size_inches(18.5, 10.5)
        plt.title("XGBoost Tree")
        plt.savefig(f"./static/data/{username}/images/{method}/xgboost_tree.png")
        plt.clf()
    elif method == "svm":
        model = SVR()
        model.fit(x_train, y_train)
    elif method == "voting":
        reg1 = GradientBoostingRegressor(random_state=1)
        reg2 = RandomForestRegressor(random_state=1)
        reg3 = LinearRegression()
        model = VotingRegressor(estimators=[("gb", reg1), ("rf", reg2), ("lr", reg3)])
        model.fit(x_train, y_train)
    # elif method == 'lightgbm':
    #     num_round = 10
    #     param = {'num_leaves': 31, 'objective': 'binary'}
    #     model = lgb.train(param, y_train, num_round, valid_sets=[x_train])
    elif method == "catboost":
        model = CatBoostRegressor(
            iterations=2,
            learning_rate=1,
            depth=2,
            loss_function="RMSE",
            verbose=None,
            allow_writing_files=False,
        )
        model.fit(x_train, y_train)
    # elif method == 'auto_sklearn':
    #     model = autosklearn.regression.AutoSklearnRegressor(
    #         time_left_for_this_task=120,
    #         per_run_time_limit=30,
    #         tmp_folder='./tmp/autosklearn_regression_tmp',
    #     )
    #     model.fit(x_train, y_train)

    # 在测试集上预测
    y_pred = model.predict(x_test)

    # format function returns a string
    mape = format(mean_absolute_percentage_error(y_test, y_pred), ".2f")
    mae = format(mean_absolute_error(y_test, y_pred), ".2f")
    mse = format(mean_squared_error(y_test, y_pred), ".2f")
    rmse = format(math.sqrt(float(mse)), ".2f")
    r2 = format(r2_score(y_test, y_pred), ".2f")

    res = [
        {"indicator": "MAPE", "value": mape},
        {"indicator": "MAE", "value": mae},
        {"indicator": "MSE", "value": mse},
        {"indicator": "RMSE", "value": rmse},
        {"indicator": "R-squared", "value": r2},
    ]

    accuracy_res = calculate_regression_accuracy(np.squeeze(y_test.values), y_pred)
    res.extend(accuracy_res)

    return res


def calculate_regression_accuracy(y, y_pred):  # gt & predicted value
    count_percent = [0, 0, 0, 0]  # 5%, 10%, 15%, 20%
    assert len(y.shape) == len(y_pred.shape)
    data_length = y.shape[0]  # y is the GT
    for i in range(data_length):
        accuracy = (y_pred[i] - y[i]) / y[i]
        for j in range(4):
            if accuracy <= (j + 1) * 0.05:
                count_percent[j] += 1
    count_percent = [(i / data_length) for i in count_percent]

    res = [
        {"indicator": "5%准确率", "value": format(count_percent[0], ".2f")},
        {"indicator": "10%准确率", "value": format(count_percent[1], ".2f")},
        {"indicator": "15%准确率", "value": format(count_percent[2], ".2f")},
        {"indicator": "20%准确率", "value": format(count_percent[3], ".2f")},
    ]

    return res


@router.post("/classification/train")
def train_classification_model(
    username: str = Form(...), percent: float = Form(...), method: str = Form(...)
):
    # 读数据文件
    df = pd.read_csv(f"./static/data/{username}/data_zscore_fill_filter.csv")

    # 划分训练集和测试集
    x = df.iloc[:, 1:]
    y = df.iloc[:, :1]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=percent, random_state=42
    )

    # 选择模型并拟合
    if method == "decision_tree":
        model = DecisionTreeClassifier()
        model.fit(x_train, y_train)
    elif method == "adaboost":
        model = AdaBoostClassifier(n_estimators=100)
        model.fit(x_train, y_train)
    elif method == "naive_bayes":
        model = GaussianNB()
        model.fit(x_train, y_train)
    # elif method == 'auto_sklearn':
    #     model = autosklearn.classification.AutoSklearnClassifier(
    #         time_left_for_this_task=120,
    #         per_run_time_limit=30,
    #         tmp_folder='./tmp/autosklearn_classification_tmp',
    #     )
    #     model.fit(x_train, y_train)

    # 在测试集上预测
    y_pred = model.predict(x_test)

    # format function returns a string
    accuracy = format(accuracy_score(y_test, y_pred), ".2f")
    precision = format(precision_score(y_test, y_pred), ".2f")
    recall = format(recall_score(y_test, y_pred), ".2f")
    auroc_score = format(roc_auc_score(y_test, y_pred), ".2f")
    auprc_score = format(average_precision_score(y_test, y_pred), ".2f")
    f1_score_result = format(f1_score(y_test, y_pred), ".2f")

    res = [
        {"indicator": "accuracy", "value": accuracy},
        {"indicator": "precision", "value": precision},
        {"indicator": "recall", "value": recall},
        {"indicator": "auroc_score", "value": auroc_score},
        {"indicator": "auprc_score", "value": auprc_score},
        {"indicator": "f1_score", "value": f1_score_result},
    ]
    pathlib.Path(f"./static/data/{username}/images/{method}").mkdir(
        parents=True, exist_ok=True
    )
    auroc_curve = plot_roc_curve(model, x_test, y_test)
    plt.title(f"{method}_auroc_curve")
    plt.savefig(f"./static/data/{username}/images/{method}/auroc.png")
    plt.clf()
    # print(res)
    return res


@router.post("/autogluon/train")
def train_autogluon(username: str = Form(...), percent: float = Form(...)):
    df = pd.read_csv(f"./static/data/{username}/data.csv")
    cols = df.columns.tolist()
    label = cols[0]
    # 划分训练集和测试集
    # x = df.iloc[:, 1:]
    # y = df.iloc[:, :1]
    train_data, test_data = train_test_split(df, test_size=percent, random_state=42)
    save_path = f"./static/data/{username}/autogluon"

    # train
    predictor = TabularPredictor(label=label, path=save_path).fit(train_data)

    # test
    y_test = test_data[label]
    test_data_nolab = test_data.drop(columns=[label])
    y_pred = predictor.predict(test_data_nolab)

    # save evaluation scores
    perf = predictor.evaluate_predictions(
        y_true=y_test, y_pred=y_pred, auxiliary_metrics=True
    )
    res = []
    for k, v in perf.items():
        res.append({"indicator": k, "value": round(v, 2)})
    return res


@router.post("/autogluon/predict")
def predict_autogluon(username: str = Form(...)):
    df = pd.read_csv(f"./static/data/{username}/data.csv")

    save_path = f"./static/data/{username}/autogluon"
    predictor = TabularPredictor.load(save_path)

    # predict
    y_pred = predictor.predict(df)
    df = pd.concat([pd.DataFrame({"label": y_pred}), df], axis=1)
    df.to_csv(f"./static/data/{username}/data_pred.csv", index=False)
    res = {"message": "success"}

    return res
