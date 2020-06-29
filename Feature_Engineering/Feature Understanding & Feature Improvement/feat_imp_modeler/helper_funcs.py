import numpy as np
import pandas as pd
import sys
sys.path.append('./modeler/')
import viz as vz
from collections import defaultdict
import math
import ppl as pl

def train_models(train_dfs,orig_rows,pred_col,data_dict_df):
        
        remove_collinear=[True,False,None]
        model_dict=defaultdict(lambda:'')
        for train_df_orig,name_df,feat_cols in train_dfs:


            #test each against multicollinearity removed and not removed
            for remove in remove_collinear:
                train_df=train_df_orig.copy()

                feat_cols=train_df.drop(pred_col,axis=1).columns


                if remove is not None:

                    #print("Columns Before Dropping Multicollinear : {}".format(train_df.shape[1]))

                    train_df=correlation(train_df,0.3,"hospital_death",remove)
                    feat_cols=train_df.drop(pred_col,axis=1).columns
                    #print("Columns After Dropping Multicollinear : {}".format(len(feat_cols)))

                non_missing_df=train_df.dropna() #remove nans first BEFORE casting to datatypes
                non_missing_df_fixed=map_loaded_default(data_dict_df,non_missing_df) #after dropping missing values, map columns dtype to original dtype
                non_missing_df_fixed_unique=non_missing_df_fixed.drop_duplicates(subset=feat_cols,keep=False)

                cat_feats,bool_feats,num_feats=desc_feats(non_missing_df_fixed_unique.drop(pred_col,axis=1))


                rows=non_missing_df_fixed_unique.shape[0]
                rows_pct=np.abs((non_missing_df_fixed_unique.shape[0]/orig_rows)*100)
                rows_pct=float("{:.2f}".format(rows_pct))

                #k_list=[k for k in range(int(math.sqrt(rows)/2),1,-11) if k%2!=0]
                #print("Original Columns : ",non_missing_df_fixed_unique.shape)
                modeler=pl.ModelerPipeline(non_missing_df_fixed_unique,pred_col,num_feats,cat_feats)
                
                modeler.fit() #fit the model
                
                coll=" Collinearity Retained"
                if remove: coll=" Collinearity Removed Without Highest Correlation"
                elif remove==False: coll=" Collinearity Removed Randomnly" 

                if name_df=="Previous Iteration Columns LR":

                    coll=name_df

                title=name_df+coll
                error,y_preds,y_scores=modeler.predict(plot=False)
                model_dict[title]={"model":modeler,"error":error,"y_preds":y_preds,\
                           "y_scores":y_scores,"feats":feat_cols,\
                          "original_rows":train_df.shape,\
                          "unique_rows":non_missing_df_fixed_unique.shape,\
                          "pct":int((rows/orig_rows)*100),\
                          "name":title}

                """print("Stats For : {}".format(title))
                print("Original Rows & Columns : {} ".format(orig_rows))
                print("Unique Rows & Columns: {} ".format(orig_rows))
                print("% of Original Dataset For Training & Testing : {}% ".format(int((rows/df.shape[0])*100)))"""
        
        model_df=pd.DataFrame.from_dict(model_dict).T
        model_df["model"]=model_df.index
        title="ROC_AUC Distribution of Unfixed & Fixed Data"
        vz.count_plot(model_df,y="error",x="model",matplotlib_use=False,title=title,showlegend=False)

        best_model=model_df.iloc[model_df.error.values.argmax()]
        print("Stats For Best Model...... : {}".format(best_model["name"]))
        print("AUC_ROC : {} ".format(best_model["error"]))
        print("Original Rows & Columns : {} ".format(best_model["original_rows"]))
        print("Unique Rows & Columns: {} ".format(best_model["unique_rows"]))
        print("% of Original Dataset For Training & Testing : {}% ".format(best_model["pct"]))      
        
        return model_dict,model_df


def correlation(dataset,threshold,label_col,check_label=None):
    
    if threshold==0: return dataset
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr().abs()
    for i in range(len(corr_matrix.columns)):
        
        
        
        parent_col=corr_matrix.columns[i]
        
        if  parent_col==label_col: continue
        
        #print("Parent Col : {}".format(parent_col))
        
        for j in range(i):
            
            #print(i,j)
            #print(corr_matrix.iloc[i, j] )
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                
                colname = corr_matrix.columns[j] # getting the name of column that has threshold correlation with this variable
                
                #print("colname : {}".format(colname),"*****************\n")
                
                if check_label is not None:
                    
                    col_names=[parent_col,colname]
                    corrs=[corr_matrix.loc[label_col][parent_col],corr_matrix.loc[label_col][colname]]

                    colname=col_names[np.argmin(corrs)]
                    #print("Deleting : ",colname)
                    #print(col_names,corrs)
                    #col_corr.add(colname)
                if colname in dataset.columns:
                    #print("Deleting from dataset",to_keep)
                    del dataset[colname] # deleting the column from the dataset

    return dataset



def find_zero_outliers(out_df,zero_columns,cat_col="hospital_death",remove_out=True):

    temp_df=pd.DataFrame()
    
    df_outlier=out_df.copy()
    
    for col in zero_columns:
        
        if remove_out:
            
            cleaned_df,df_outlier=remove_outlier(out_df,col) #get outliers of this variable
            
            
           
        grouped_df_out=df_outlier[df_outlier[col]==0].groupby([cat_col]).count()[col]#[cat_col].value_counts()
        grouped_df_out=pd.DataFrame(grouped_df_out).T
        
        temp_df=temp_df.append(grouped_df_out)

    temp_df[cat_col]=temp_df.index
    temp_df.reset_index(inplace=True,drop=True)
   
    return temp_df.fillna(0)

def remove_outlier(df_in, col_name,c=1.5):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-c*iqr
    fence_high = q3+c*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    out = df_in.loc[(df_in[col_name] < fence_low) | (df_in[col_name] > fence_high)]
    return df_out,out


        
from collections import defaultdict
def check_missing_string(df,cols):
    
    cat_list=defaultdict(lambda:'')
    
    for col in cols:
        
        print("\nCol : {}".format(col))
    
        string_cols=df[col].apply(lambda arg: arg.strip()==" ")
        cat_list[col]=np.sum(string_cols)
        value_counts=df[col].value_counts()
        print("Total Sum of Type of Values Present : {} & Total Df Shape : {}".format(np.sum(value_counts),df.shape))
        print("Categories : \n{}".format(value_counts))
        
    return cat_list        
    
    

def check_zeros(df,data_dict_df,cat_col_list): # check for zero for a given category
    
    zeros_dict=defaultdict(lambda:'')
    for cat_col in cat_col_list:
        
        present=False
        ### check desriptions of pure numerical categories which included
        variable_names=data_dict_df[data_dict_df["Category"]==cat_col]["Variable Name"].tolist()#.plot(kind="line")
        
        for var in variable_names:
            
            zeros=np.sum(df[var].sort_values()==0)
            
            if zeros>0:
                zeros_dict[var]=zeros
                present=True
                print("Total Zero Values For Variable {} in Category '{}' : {} ".format(var,cat_col,zeros))

        if present==False:

            print("\nNone of the variables in category : '{}' has any zero values".format(cat_col))
     
    return zeros_dict
    
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