import shap
from sklearn.ensemble import RandomForestClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from pycaret.classification import *
import optuna
import Reqs
import numpy as np
from sklearn.metrics import accuracy_score
import databricks.sql
from sqlalchemy import create_engine
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
import ML_Model_Preprocessing_Loading as MLMPL
import matplotlib.pyplot as plt

def Create_Model(data, target, model_selection, ignore_features):
    clf1 = setup(data = data, target = target, fix_imbalance = True,remove_multicollinearity=True, ignore_features=ignore_features,feature_selection = True,n_features_to_select=0.6)
    if model_selection == 'compare':
        best = compare_models()
    else:
        try:
            best = create_model(model_selection)
        except:
            print(f'Error creating model: {model_selection}, please select model from list and try again')
            models()
    evaluate_model(best)
    return clf1, best