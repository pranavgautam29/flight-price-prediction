# import essential modules and packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


# load the data
def clean_data(data_path):

    # get the data using read_csv
    df = pd.read_csv(data_path)

    # drop unnecessary features
    df = df.drop(columns="Unnamed: 0")

    # drop the features with high cardinality
    cardinality_threshold = 100
    high_cardinality_feature = [col for col in list(df.select_dtypes("object").columns) if
                                df[col].nunique() > cardinality_threshold]
    df = df.drop(high_cardinality_feature, axis=1)

    return df

