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

def Model_Generate_Shap_Diagrams(model):
    interpret_model(model, plot='summary')
    interpret_model(model, plot='correlation')
    interpret_model(model, plot='msa')