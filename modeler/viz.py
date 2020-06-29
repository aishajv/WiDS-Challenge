import pandas as pd
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt   
import seaborn as sns 
import plotly.express as px
import plotly.io as pio
from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot, plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_pie(df,col,title,legend_names=None,value_counts=True):
    
    if value_counts:
        counts=df[col].value_counts().values
        values=df[col].value_counts().index
        plt.pie(counts, labels=values,autopct='%1.2f%%')
        
    else:
        pie_values=df[col]
        pie_labels=df[legend_names]
        plt.pie(pie_values, labels=pie_labels,autopct='%1.2f%%')


def plot_roc_auc(fpr,tpr,roc_auc,clf="KNN"):
    
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve of '+clf)
    plt.show()


def viz_corr(corr_matrix,label="hospital_death",thresh=0.1,title="Correlation of Columns with Label : ",figsize=(20,10)):
    
    title+=label+" Thresh : "+str(thresh)
    
    label_corr=corr_matrix.loc[label].sort_values()
    count_plot(label_corr[label_corr>=thresh],x="Columns",y="Corr Value",matplotlib_use=False,title=title,figsize=figsize)
    
"""

def line_plot(series,x,y):
   
    df=pd.DataFrame(data=np.column_stack([series.index,series.values]),columns=[x,y])
    fig = px.line(df, x=x, y=y, title='Life expectancy in Canada')
    fig.show()
"""
def count_plot(series=None,title=None,x=None,y=None,matplotlib_use=False,kind="barh",figsize=(50,10)):
    
    df=series
    if isinstance(series,pd.Series):
        df=pd.DataFrame(data=np.column_stack([series.index,series.values]),columns=[x,y])

    
    plt.figure(figsize=figsize)
    
    if not matplotlib_use:
        
        fig = px.bar(df, y=y, x=x, color=x)
        fig.update_traces(texttemplate='%{text:.1s}', textposition='none')
        fig.update_layout(uniformtext_minsize=5, uniformtext_mode='hide',title=title)
        fig.show()
       
    else:
       
        n = df[x].unique().__len__()+1
        all_colors = list(plt.cm.colors.cnames.keys())
        c = random.choices(all_colors, k=n)

            
        # Plot Bars
        plt.figure(figsize=figsize, dpi= 80)
        df[x]=df[x].astype(str)
        plt.bar(df[x], df[y], color=c, width=.5)
        for i, val in enumerate(df[y].values):
            plt.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight':500, 'size':8})

        # Decoration
        plt.gca().set_xticklabels(df[x], rotation=60, horizontalalignment= 'right')
        plt.title(title, fontsize=8,verticalalignment='bottom')
        plt.ylabel(y)
        plt.ylim(0, max(df[y]))
        plt.show()
        
    

"""
def plot_change(df1,df2,custom_func):
  	
  
  	#df1 refers to orignal ALWAYS
    numeric_cols=df1.select_dtypes(include=np.number).columns.tolist()

    #perform custom functions on both dfs
    df1_custom_func=np.array([custom_func(df1[col]) for col in df1[numeric_cols]])
    df2_custom_func=np.array([custom_func(df2[col]) for col in df2[numeric_cols]])

    #compute diffs
    custom_func_diffs=np.abs((df1_custom_func - df2_custom_func) / df1_custom_func)
    custom_func_diffs=custom_func_diffs*100
    plt.bar(numeric_cols,custom_func_diffs)
    plt.title('% change in average column values')
    plt.ylabel('% change')
    plt.show()
    
    
def plot_boxplot(df,rows,cols,fig_size=(15,15)):
  fig=plt.figure(figsize=fig_size)
  
  
  for col_idx,col in enumerate(df.select_dtypes(include=np.number)):
       
      
      if(col_idx+1>(rows+cols+1)):break
      ax = fig.add_subplot(rows,cols,col_idx+1)
      df.boxplot(col,ax=ax)
      ax.set_title(col)
   

  plt.show()
"""

