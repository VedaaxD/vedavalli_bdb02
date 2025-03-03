#partitioning the dataset for decision trees
import numpy as np
import pandas as pd
file_path='/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv'
df=pd.read_csv(file_path)

def partition_dataset(df,threshold):
    df_low=df[df['BP']<=threshold]
    df_high=df[df['BP']>threshold]
    return df_low,df_high

threshold=[80,78,82]
for i in threshold:
    df_low,df_high=partition_dataset(df,i)
    print(f"Partitioned dataset < {i} :\n{df_low}")
    print(f"Partitioned dataset > {i} :\n{df_high}")