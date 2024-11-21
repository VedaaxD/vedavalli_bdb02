import numpy as np
import pandas as pd
def load_file(path):
    try:
        titanic=pd.read_csv('/home/ibab/Downloads/titanic.csv')
        return titanic
    except FileNotFoundError as f:
        raise FileNotFoundError(f"File path hasn't been found.")
    except Exception as e:
        raise Exception(f"Error has occured.")
def total_rows(titanic):
    return titanic.shape[0]
def total_columns(titanic):
    return titanic.shape[1]
def fill_missing_val(titanic):
    return titanic.fillna(0)