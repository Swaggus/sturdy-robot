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

def DRP_Agena_Ingestion_Model_Preprocess(data):
    """This preprocesses data for the DRP Ingestion Model, Must provide the data in format, use fetch data to do so."""
    car_dim = pd.read_csv('Car_dim.csv')
    data = pd.merge(data, car_dim, left_on='Make', right_on='Combined Make and Model', how='right')
    data['Outward_Code'] = data['Postcode'].str.extract(r'([A-Z]{1,2}[0-9R][0-9A-Z]?)')
    print('Outward code generated...')
    data['Inward_Code'] = data['Postcode'].str.extract(r'(\d[A-Z]{2})$')
    print('Inward code generated...')
    data['ShortPostcode'] = data['Postcode'].str.extract(r'([A-Za-z]+)')
    print('Short Postcode generated...')
    Postcode_Table = pd.DataFrame(data.ShortPostcode.unique())
    data['OffenceDate'] = pd.to_datetime(data['OffenceDate'])
    print('Offence Date Dtype converted to DateTime...')
    data = data.dropna(subset=['OffenceTime'])
    data['OffenceTime'] = data['OffenceTime'].str[2:4]
    print('Offense Time Hour Extracted...')
    data['OffenceTime'] = data['OffenceTime'].astype(int)
    print('Offense Time Hour converted to Int...')
    bins = [-1, 8, 12, 17, 21, 24]
    print('Creating Time Bins...')
    bin_labels = ['Early Morning', 'Late Morning', 'Afternoon', 'Evening', 'Night']
    print('Create Time Bin Labels...')
    data['OffenceTime_binned'] = pd.cut(data['OffenceTime'], bins, labels=bin_labels)
    print('Applying Bins to Dataset...')
    data['day_of_week'] = data['OffenceDate'].dt.day_name()
    print('Generating Day of Week...')
    data['Year'] = data['OffenceDate'].dt.year
    print('Extracting Year...')
    data['Month'] = data['OffenceDate'].dt.month
    print('Extracting Month...')
    #data['Day'] = data['OffenceDate'].dt.day
    data['dof'] = pd.to_datetime(data['dof'])
    print('Converting DOF to DateTime...')
    data['Offense_Count'] = data.groupby('Registration')['Registration'].transform('count')
    print('Generating Offense Count...')
    data['AnyPaymentReceived'] = [1 if x > 0 else 0 for x in data['CR_6mGBP']]
    print('Generating Target Column...')
    def determine_target(group):
        if group['AnyPaymentReceived'].max() == 1:
            group.iloc[0, group.columns.get_loc('AnyPaymentReceived')] = 1
        return group.iloc[0]

    # Group by registration and apply the function
    print('De-Duping Registration, this may take a few minutes...')
    data = data.groupby('Registration').apply(determine_target)
    print('Generating BatchIDs')
    data['month_bt'] = data['adjdof'].astype(str).str[5:7]
    data['year'] = data['Year'].astype(str).str[:4]
    data['BatchID'] = "BT" + data['month_bt'] + data['year']
    # Reset the index
    data.reset_index(drop=True, inplace=True)
    data = data[['Debt_id','BatchID','DESCRIPTION','Make_y','Model_y','Price Point','month','Registration','age_of_debt','Outward_Code','client_name','OffenceTime_binned','day_of_week','Offense_Count','AnyPaymentReceived',]]
    print('Finalising Data...')
    features = data[['Debt_id','DESCRIPTION','Make_y','Model_y','Price Point','month','Registration','age_of_debt','Outward_Code','client_name','OffenceTime_binned','day_of_week','Offense_Count']]
    print('Finalising Features...')
    target = data['AnyPaymentReceived']
    print('Finalising Target...')
    return data, features, target

def DRP_Agena_Ingestion_Model_Fetch_Data(segment='train'):
    # Replace with your Databricks workspace details
    hostname = "adb-2877091482490491.11.azuredatabricks.net"
    http_path = "sql/protocolv1/o/2877091482490491/1018-124314-t95kgjl6"
    token = Reqs.PersonalToken

    # Create a connection
    connection = databricks.sql.connect(server_hostname=hostname,
                                    http_path=http_path,
                                    access_token=token)

    # Execute a query and fetch the results
    with connection.cursor() as cursor:
        cursor.execute(f'SELECT * FROM hive_metastore.default.agena_model{segment}_data_dec2023_rjc')
        # Fetch the results into a list
        rows = cursor.fetchall()

        # Assuming you know the column names and types
        column_names = [column[0] for column in cursor.description]
        df = pd.DataFrame(rows, columns=column_names)

    # Close the connection
    connection.close()
    data = df
    return data