import os,csv,json

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, Form, File, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from dowhy import CausalModel

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from IPython.display import Image, display

import app.schemas as schemas
import app.crud as crud
from app.database import SessionLocal

# -*- coding: UTF-8 -*-

router = APIRouter(prefix = "/causals")

@router.post("/{data_filename}/model_build")
async def return_dimension_reduction(info: schemas.CausalInfo, data_filename: str):
    df = pd.read_csv(f"./static/data/{data_filename}_zscore_fill_filter.csv")
    causal_graph = """
        digraph {"""+ info.key + """U[label="Unobserved Confounders"];"""+info.causal + """}"""
    model= CausalModel(
        data = df,
        graph=causal_graph.replace("\n", " "),
        treatment='High_limit',
        outcome='Churn')
    model.view_model()  
    estimands = model.identify_effect()
    estimate = model.estimate_effect(estimands,method_name = "backdoor.propensity_score_weighting")
    refutel_1 = model.refute_estimate(estimands,estimate, "random_common_cause")
    refutel_2 = model.refute_estimate(estimands,estimate, "data_subset_refuter")
    refutel_3 = model.refute_estimate(estimands,estimate, "placebo_treatment_refuter")
    response={
        'estimands': estimands,
        'estimate': estimate,
        'refute_r': refutel_1,
        'refute_d': refutel_2,
        'refute_p': refutel_3,
     }
    return response