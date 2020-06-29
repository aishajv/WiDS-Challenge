import numpy as np
import pandas as pd


def desc_feats(df):

    cols=df.columns
    bool_feats=list(df.select_dtypes(include=bool).columns)
    numeric_feats=df._get_numeric_data().columns
    cat_feats=list(set(cols)-set(numeric_feats))
    numeric_feats=set(numeric_feats)-set(bool_feats)
    
    return list(cat_feats),list(bool_feats),list(numeric_feats)

def print_dtypes(df):
    dtypes_series=df.dtypes
    for index,elem in zip(dtypes_series.index,dtypes_series):
        print(index,elem)
    print("\n")

def print_cols(df):
    for col in df.columns: print(col)
    print("\n")

def get_miss_non_miss_df(df):
    
    missing_df=df.isnull().sum()
    non_missing_columns=missing_df[missing_df==0].index
    non_missing_df=df[non_missing_columns]
    missing_columns=missing_df[missing_df>0].index
    missing_df_only=df[missing_columns]
    non_missing_columns=missing_df[missing_df==0].index
    non_missing_df=df[non_missing_columns]
    missing_columns=missing_df[missing_df>0].index
    missing_df_only=df[missing_columns]
    
    return missing_df_only,non_missing_df

import numpy as np

def map_loaded_default(data_dict_df,df):

    df=df.copy()
    
    
    mapper={"numeric":np.float64,
            "binary":np.bool,
            "string":str,
            "integer":np.int64}
    
  
    dict_df=data_dict_df[["Variable Name","Data Type"]]
    dict_df.set_index("Variable Name",inplace=True)
    dict_df=dict_df.T.to_dict("records")[0]
    dict_df={k.lower(): v for k, v in dict_df.items()}
    df.columns=map(str.lower, df.columns)
  
    for col in df.columns:
       
        try:
            df[col]=df[col].astype(mapper[dict_df[col]])
        
        except:
            
            print("Column Name : {}".format(col))
            print("Data Dict Encoding Data Type : ",dict_df[col])
            print("Failed Encoded Data Type : {}".format(mapper[dict_df[col]]))
            
            df[col]=df[col].astype(mapper["numeric"])
            print("Successfully Encoded Data Type : {}\n******************************".format(df[col].dtype))
        

    return df