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
import sys
sys.path.append('./feat_imp_modeler/')
import helper_funcs as hf
from collections import defaultdict
import cufflinks as cf



def plot_multiple_plotly(df,df_vars,cat_col,**kwargs):
    
    rows=kwargs.get("rows",None)
    cols=kwargs.get("cols",None)
    type_plot=kwargs.get("plot_type","hist")
    title=kwargs.get("title","")
    barmode=kwargs.get("barmode",None)
        
    if barmode is not None:   
        
        if type_plot=="bar": #useful when all plot can be plotted on a single graph

            name=kwargs.get("label","hospital_death")


            x=df[cat_col]

            fig = go.Figure()

            for var in df_vars:

                fig.add_trace(go.Bar(x=x, y=df[var], name=name+" : "+str(var)))

        #fig.update_layout(barmode='stack', title=title)
        #fig.show()
        
        

    
    else:
        stack_plot=kwargs.get("stack",True)
        subplot_titles=kwargs.get("subplot_titles",df_vars)

        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles)

        df_index=-1
        len_counter=0
        for row_idx in range(rows):

            for col_idx in range(cols):

                #len_counter+=1
                df_index+=1

                if df_index+1>len(df_vars):break

                df_var=df_vars[df_index]


                if type_plot=="bar_outliers":

                        cleaned_df,df_outlier=hf.remove_outlier(df,df_var) #get outliers of this variable
                        gr_df=pd.DataFrame(df_outlier.groupby([df_var]).count()[cat_col])
                        gr_df.reset_index(inplace=True)

                        fig.add_trace(go.Scatter(x=gr_df[df_var], y=gr_df[cat_col]/float(df_outlier.shape[0]), name=df_var),row=row_idx+1,col=col_idx+1)

                else:

                        for unique_value in df[cat_col].value_counts().index:
                            
                            if type_plot=="box":

                                fig.add_trace(go.Box(y=df[df[cat_col]==unique_value][df_var],
                                              name=cat_col+" = "+str(unique_value)),row=row_idx+1,col=col_idx+1)

                            elif type_plot=="hist":

                                nbins=kwargs.get("nbins",10)
                                histnorm=kwargs.get("histnorm",None)

                                fig.add_trace(go.Histogram(x=df[df[cat_col]==unique_value][df_var],\
                                                           name=df_var + " - "+cat_col+" = "+str(unique_value),histnorm=histnorm,nbinsx=nbins),\
                                              row=row_idx+1,col=col_idx+1)

                #fig.update_xaxes(title_text=df_var, row=row_idx, col=col_idx)
                
                

    fig.update_yaxes(title_text=kwargs.get("yaxis","counts"))
    fig.update_layout(
        
        autosize=False,
        width=kwargs.get("width",1000),
        height=kwargs.get("height",1000),
        showlegend=kwargs.get("show_legend",True),
        barmode=barmode,
        title=title)
    
    fig.show()



def plot_pie_plotly(df,x_col_pct,title):
    
    names=df[x_col_pct].value_counts().index
    values=df[x_col_pct].value_counts().values
    
    fig = px.pie(df, values=values, names=names,
                 title=title)
    fig.update_traces(textposition='inside', textinfo='percent+label',)
    fig.show()


def plot_cat_bar(df,x_values,color_cat,title,nbins=None,histnorm=None):
    fig = px.histogram(df, x=x_values, color=color_cat,title=title,histnorm=histnorm,nbins=nbins)
    fig.show()
    

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

def plot_multiples(df,label_col,columns,rows,cols,fig_size=(25,15),nbins=15,kind="hist"):
    
    fig=plt.figure(figsize=fig_size)
  
    df_cols_index=0
    
    for col_idx,sel_col in enumerate(columns):
   
            ax = fig.add_subplot(rows,cols,col_idx+1)
    
            if kind=="hist":
        
                df[df[label_col]!=0][sel_col].plot(kind=kind,x=sel_col,ax=ax,alpha=0.5,label="Hospital Death : True",bins=nbins)
                df[df[label_col]==0][sel_col].plot(kind=kind,x=sel_col,ax=ax,alpha=0.5,label="Hospital Death : False",bins=nbins)
            
            
            if kind=="box":
                df[df[label_col]!=0][sel_col].plot(kind=kind,x=sel_col,label="Hospital Death : True")
                df[df[label_col]==0][sel_col].plot(kind=kind,x=sel_col,label="Hospital Death : False")
            
            ax.set_xlabel(sel_col)
            ax.legend(loc='upper right')
            ax.set_ylabel('Frequency')
            
    plt.show()
    

"""

def line_plot(series,x,y):
   
    df=pd.DataFrame(data=np.column_stack([series.index,series.values]),columns=[x,y])
    fig = px.line(df, x=x, y=y, title='Life expectancy in Canada')
    fig.show()
"""
def count_plot(series=None,title=None,x=None,y=None,matplotlib_use=False,kind="barh",figsize=(50,10),showlegend=True):
    
    df=series
    if isinstance(df,pd.Series):
        
        df=pd.DataFrame(data=np.column_stack([series.index,series.values]),columns=[x,y])

    
    plt.figure(figsize=figsize)
    
    if not matplotlib_use:
        
        fig = px.bar(df, y=y, x=x, color=x)
        fig.update_traces(texttemplate='%{text:.1s}', textposition='none')
        fig.update_layout(uniformtext_minsize=5, uniformtext_mode='hide',title=title,showlegend=showlegend)
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
            

def plot_multiple(list_dfs,rows,cols,fig_size=(15,15),kind="bar"):
    
    
    fig=plt.figure(figsize=fig_size)

    for df in list_dfs:
        
        for col_idx,col in enumerate(df.columns):

            if(col_idx+1>(rows+cols+1)):break
            ax = fig.add_subplot(rows,cols,col_idx+1)

            if kind=="bar":
                df.plot.bar(col,ax=ax)

            ax.set_title(col)

    plt.show()



def plot_corr(filt_df,title,figsize=(15,5),scale_log=False,compute_corr=True,color_scheme="OrRd",**cbar_kws):
    
    corr_matrix=filt_df
    if compute_corr:
        corr_matrix=filt_df.corr()
    plt.figure(figsize=figsize)
    ax=sns.heatmap(corr_matrix,annot=True,cmap=color_scheme,cbar_kws=cbar_kws)
    ax.set_title(title)
    plt.show()
    return corr_matrix


def plot_missing_pct(df,all_columns):
    missing_values_series=df[all_columns].isnull().sum()
    missing_columns=missing_values_series[missing_values_series>0]
    missing_columns_df=(missing_columns.sort_values()/df.shape[0])*100
    title="Distribution of Missing Values Pct % Vs Attributes"
    x_col="Attributes"
    y_col="Missing Pct %"
    missing_columns_df=pd.DataFrame(data=np.column_stack([missing_columns_df.index,missing_columns_df.values]),columns=[x_col,y_col])
    
    count_plot(missing_columns_df,x=x_col,y=y_col,matplotlib_use=False,title=title)  
    
def plot_change(df1,df2,custom_func=None,kind="bar",title="",width=1000,height=500):
  	
  
  	#df1 refers to orignal ALWAYS
    numeric_cols=df1.select_dtypes(include=np.number).columns.tolist()

    #perform custom functions on both dfs
    if custom_func is not None:
    
        diff=pd.DataFrame((custom_func(df1)-custom_func(df2))/custom_func(df1)*100)
    else:
        diff=(df1-df2).abs()
        
   
    fig=diff.iplot(asFigure=True,kind=kind)
    fig.update_layout(title=title,width=width,height=height)
    fig.show()
    return diff
    """plt.bar(numeric_cols,custom_func_diffs)
    plt.title('% change in average column values')
    plt.ylabel('% change')
    plt.show() """  
    
"""

    
def plot_boxplot(df,rows,cols,fig_size=(15,15)):
  fig=plt.figure(figsize=fig_size)
  
  
  for col_idx,col in enumerate(df.select_dtypes(include=np.number)):
       
      
      if(col_idx+1>(rows+cols+1)):break
      ax = fig.add_subplot(rows,cols,col_idx+1)
      df.boxplot(col,ax=ax)
      ax.set_title(col)
   

  plt.show()
"""

