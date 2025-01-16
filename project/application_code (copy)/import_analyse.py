# necessary imports 
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io
import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
#%matplotlib inline
pd.set_option('display.max_columns', 26)


def basic_info(file_path):
    df = pd.read_csv(file_path)
    df.drop('Unnamed: 0', axis = 1, inplace = True)
    df.drop('Date', axis = 1, inplace = True)

   
    #df.drop('id', axis=1, inplace=True)
    df.columns = ['State', 'City', 'Crop Type', 'Season', 'Temperature (°C)',
           'Rainfall (mm)', 'Supply Volume (tons)', 'Demand Volume (tons)',
           'Transportation Cost (₹/ton)', 'Fertilizer Usage (kg/hectare)',
           'Pest Infestation (0-1)', 'Market Competition (0-1)', 'Price (₹/ton)']
    
    head_html = df.head().to_html(classes='table table-striped')
    describe_html = df.describe().to_html(classes='table table-striped')
    shape = df.shape
    info_buf = io.StringIO()
    df.info(buf=info_buf)
    info_html = info_buf.getvalue().replace('\n', '<br>')
    
    return head_html, shape, describe_html, info_html

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.drop('Unnamed: 0', axis = 1, inplace = True)
    df.drop('Date', axis = 1, inplace = True)

   
    #df.drop('id', axis=1, inplace=True)
    df.columns = ['State', 'City', 'Crop Type', 'Season', 'Temperature (°C)',
           'Rainfall (mm)', 'Supply Volume (tons)', 'Demand Volume (tons)',
           'Transportation Cost (₹/ton)', 'Fertilizer Usage (kg/hectare)',
           'Pest Infestation (0-1)', 'Market Competition (0-1)', 'Price (₹/ton)']
    
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    num_cols = [col for col in df.columns if df[col].dtype != 'object']

    num_nulls_before = df[num_cols].isnull().sum().to_dict()
    cat_nulls_before = df[cat_cols].isnull().sum().to_dict()

    def random_value_imputation(feature):
        random_sample = df[feature].dropna().sample(df[feature].isna().sum())
        random_sample.index = df[df[feature].isnull()].index
        df.loc[df[feature].isnull(), feature] = random_sample

    def impute_mode(feature):
        mode = df[feature].mode()[0]
        df[feature] = df[feature].fillna(mode)

    for col in num_cols:
        random_value_imputation(col)

    random_value_imputation('Rainfall (mm)')
    random_value_imputation('Temperature (°C)')

    for col in cat_cols:
        impute_mode(col)

    num_nulls_after = df[num_cols].isnull().sum().to_dict()
    cat_nulls_after = df[cat_cols].isnull().sum().to_dict()

    return (num_nulls_before, cat_nulls_before, num_nulls_after, cat_nulls_after, df.head().to_html(classes='table table-striped'))
def violin_plot(col):
    fig = px.violin(df, y=col, x="Price (₹/ton)", color="Price (₹/ton)", box=True, template='plotly_dark')
    return fig.to_html(full_html=False)

def kde_plot(col):
    plt.figure(figsize=(10, 6))
    grid = sns.FacetGrid(df, hue="Price (₹/ton)", height=6, aspect=2)
    grid.map(sns.kdeplot, col)
    grid.add_legend()
    plt.tight_layout()
    return plt.gcf().canvas.get_supported_filetypes()['png']

def scatter_plot(col1, col2):
    fig = px.scatter(df, x=col1, y=col2, color="Price (₹/ton)", template='plotly_dark')
    return fig.to_html(full_html=False)

def eda_plots(file_path):
    df = pd.read_csv(file_path)
    df.drop('Unnamed: 0', axis = 1, inplace = True)
    df.drop('Date', axis = 1, inplace = True)

   
    #df.drop('id', axis=1, inplace=True)
    df.columns = ['State', 'City', 'Crop Type', 'Season', 'Temperature (°C)',
           'Rainfall (mm)', 'Supply Volume (tons)', 'Demand Volume (tons)',
           'Transportation Cost (₹/ton)', 'Fertilizer Usage (kg/hectare)',
           'Pest Infestation (0-1)', 'Market Competition (0-1)', 'Price (₹/ton)']
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    num_cols = [col for col in df.columns if df[col].dtype != 'object']
    # Numerical features distribution
    plt.figure(figsize=(20, 15))
    plotnumber = 1
    for column in num_cols:
        if plotnumber <= 14:
            ax = plt.subplot(3, 5, plotnumber)
            sns.distplot(df[column])
            plt.xlabel(column)
        plotnumber += 1
    plt.tight_layout()
    plt.savefig('static/images/numerical_distribution.png')  # Save the plot as an image
    plt.close()

    # Categorical columns count plot
    plt.figure(figsize=(20, 15))
    plotnumber = 1
    for column in cat_cols:
        if plotnumber <= 11:
            ax = plt.subplot(3, 4, plotnumber)
            sns.countplot(df[column], palette='rocket')
            plt.xlabel(column)
        plotnumber += 1
    plt.tight_layout()
    plt.savefig('static/images/categorical_counts.png')  # Save the plot as an image
    plt.close()

    # Heatmap of data
    plt.figure(figsize=(15, 8))
    sns.heatmap(df.corr(), annot=True, linewidths=2, linecolor='lightgrey')
    plt.savefig('static/images/heatmap.png')  # Save the plot as an image
    plt.close()


