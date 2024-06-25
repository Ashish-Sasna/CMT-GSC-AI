import re
import os
import glob
import seaborn as sns
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats.mstats import winsorize
from scipy.interpolate import griddata

from sklearn.cluster import KMeans

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectKBest, f_regression

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, GlobalAveragePooling1D, Dense, Dropout, MaxPool1D, Flatten
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay

import xgboost

import pickle as pkl

import netron

from openpyxl import load_workbook

## Extracting list of chemical elements
def elem_list(df):
    str_elem = []
    
    for col in list(df.columns):
        if col not in ['gid', 
                       'objectid', 
                       'sampleno', 
                       'longitude', 
                       'latitude', 
                       'toposheet']:
            str_elem.append(col)

    return str_elem

## Element concentration variation
def plot_ppm_variation(df, element, area):
    lat_list = np.sort(df['latitude'].unique())[::-1]
    
    for lat in lat_list:
        subset = df[df['latitude'] == lat].sort_values(by='longitude')
        if area == 'Ramagiri':
            plt.figure(figsize=(10, 2))
        elif area == 'Kodangal':
            plt.figure(figsize=(12, 2))
        plt.plot(subset['longitude'], subset[element], marker='o', linestyle='-', color='b')
        plt.title(f'Concentration at Latitude {lat}')
        plt.xlabel('Longitude')
        plt.ylabel(f'Concentration(ppm)')
        plt.ylim(0, max(df[element]))
        plt.grid(True)
        plt.show()

## Contour maps
def plot_contour(df, element, name, title, area):
    # Check if the element exists in the dataframe
    if element not in df.columns:
        raise KeyError(f"The element '{element}' does not exist in the dataframe.")

    min_lat_limit = min(df['latitude']) - 0.005
    max_lat_limit = max(df['latitude']) + 0.005
    min_long_limit = min(df['longitude']) - 0.005 
    max_long_limit = max(df['longitude']) + 0.005

    grid_x, grid_y = np.mgrid[
        min_long_limit:max_long_limit:200j,  # 200j specifies 200 points in grid
        min_lat_limit:max_lat_limit:200j
    ]

    # Grid interpolation
    grid_z = griddata(
        (df['longitude'], df['latitude']),
        df[element],
        (grid_x, grid_y),
        method='cubic'  # 'cubic' for smoother contour lines
    )

    cmap = LinearSegmentedColormap.from_list("green_to_red", ["green", "yellow", "red"])
    
    if area == 'Ramagiri':
        plt.figure(figsize=(10, 6))
    elif area == 'Kodangal':
 	    plt.figure(figsize=(15, 5))

    # Filled contour
    cp = plt.contourf(grid_x, grid_y, grid_z, levels=15, cmap=cmap, alpha=0.7)
    plt.colorbar(cp, label=f'{name} concentration')

    # Line contours
    cs = plt.contour(grid_x, grid_y, grid_z, levels=15, colors='k', linewidths=0.5)
    plt.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')

    # plt.scatter(df['longitude'], df['latitude'], color='black', s=10)  # Sample points
    
    # Annotate ppm values at each sample point
    # for x, y, ppm in zip(df['longitude'], df['latitude'], df[element]):
    #     plt.text(x, y, f'{ppm:.1f}', color='black', fontsize=8, ha='right')

    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    
    # Positioning the legend outside the plot area with spacing
    plt.legend(['Sample Points'], loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

## Nearest neighbor capping
def nn_cap(elem, elem_df, df, val):

    # Assuming you define 'neighbors' based on some logical geographical proximity
    nbrs = NearestNeighbors(n_neighbors=val, 
                            algorithm='ball_tree').fit(df[['latitude', 
                                                           'longitude']])
    distances, indices = nbrs.kneighbors(df[['latitude', 
                                             'longitude']])
    
    
    for index, row in elem_df.iterrows():
        # Calculate the mean of the nearest neighbors excluding the outlier itself
        neighbor_indices = indices[index][1:]  # exclude the first index since it's the point itself
        mean = df.iloc[neighbor_indices][elem].mean()
        df.at[index, elem] = np.round(mean, 2)

    return df

## Outlier handling
def handle_outl(df, elem_list, ex_elem):

    cols = elem_list
    cols = [x for x in cols if x not in ex_elem]
    
    for elem in cols:
        
        Q1 = df[elem].quantile(0.25)
        Q3 = df[elem].quantile(0.75)
        IQR = Q3 - Q1
        lwr_bnd = Q1 - (1.5 * IQR)
        upr_bnd = Q3 + (1.5 * IQR)

        quantiles = {
            87 : df[elem].quantile(0.87), 
            90 : df[elem].quantile(0.90),
            92.5 : df[elem].quantile(0.925),
            95 : df[elem].quantile(0.95),
            97.5 : df[elem].quantile(0.975),
            99 : df[elem].quantile(0.99)
        }
        
        min_diff = float('inf')
        nearest_key = None
        for key, val in quantiles.items():
        
            if val >= upr_bnd:
                diff = val - upr_bnd
                if diff < min_diff:
                    min_diff = diff
                    nearest_key = key
        
        if nearest_key is None:
            df[elem] = df[elem].apply(lambda x: upr_bnd if x > upr_bnd else x)
            continue
        
        nearest_key = (100 - nearest_key) / 100
        nearest_key = np.round(nearest_key, 3)
        df[elem] = winsorize(df[elem], limits=(0.05, nearest_key))

    return df

## Recursive Feature Elimination with Cross-Validation
def rfecv(estimator, X, y, step, cv):
    rfecv = RFECV(estimator=estimator, 
                  step=step, 
                  cv=KFold(cv, 
                           shuffle=True, 
                           random_state=42),
                  scoring='neg_mean_absolute_error')
    rfecv.fit(X, y)
    return rfecv

## RFECV for classification
def rfecv_cls(estimator, X, y, step, cv):
    rfecv = RFECV(estimator=estimator, 
                  step=step, 
                  cv=StratifiedKFold(cv), 
                  scoring='accuracy')
    rfecv.fit(X, target)
    return rfecv

## Plotting % of Correct Classification
def plot_pcc(rfecv, title):
    
    plt.figure(figsize=(10, 9))
    plt.title(title, 
              fontsize=18, 
              fontweight='bold', 
              pad=20)
    plt.xlabel('Number of features selected', 
               fontsize=14, 
               labelpad=20)
    plt.ylabel('% Correct Classification', 
               fontsize=14, 
               labelpad=20)
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), 
             rfecv.cv_results_['mean_test_score'], 
             color='#303F9F', 
             linewidth=3)
    
    plt.show()

## Plotting feature importances
def feature_importance(rfecv, X, title): 
        
    df = pd.DataFrame()
    
    df['column'] = X.columns
    df['importance'] = rfecv.estimator_.feature_importances_
    
    df.sort_values(by='importance', 
                   ascending=False, 
                   inplace=True, 
                   ignore_index=True)
    
    sns.set(rc = {'figure.figsize':(8,10)})
    ax = sns.barplot(y='column', 
                     x='importance',
                     data=df,
                     palette='viridis')
    
    ax.set_title(title, 
                 fontsize=18)

## Predicting validation data
def pred_val(df, sc, model):

    scaled_df = sc.transform(df)
    
    scaled_df = pd.DataFrame(scaled_df, 
                             columns=df.columns)
    
    # scaled_df = scaled_df[columns]

    # pca_df = pca.transform(scaled_df)

    y_pred = model.predict(scaled_df)

    return y_pred

## Evaluate the final model
def eval_model(y, y_pred, title, n_sampl, n_obsv):
    
    mse = mean_squared_error(y, y_pred)
    medae = median_absolute_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r_squared = r2_score(y, y_pred)
    adj_r_squared = 1 - ((1 - r_squared) * (n_sampl - 1)/(n_sampl - n_obsv - 1))

    print(f'{title}')
    # print(f'Median Absolute Error: {np.round(medae, 2)}')
    print(f'Root Mean Squared Error: {np.round(rmse, 2)}')
    print(f'Mean Absolute Error: {np.round(mae, 2)}')
    print(f'R-squared Error: {np.round(r_squared, 2)}')
    print(f'Adjusted R-squared Error: {np.round(adj_r_squared, 2)}')

## Plotting Learning curves
def plot_learing_curve(model, X, y, cv, model_title, area):

    if area == 'Kodangal':
        train_sizes = [50, 100, 200, 300, 380]
    elif area == 'Ramagiri':
        train_sizes = [30, 50, 70, 100, 128]

    train_sizes, train_scores, validation_scores = learning_curve(model, X, y, 
                                                                  train_sizes=train_sizes, 
                                                                  cv=cv, 
                                                                  scoring='neg_mean_absolute_error', 
                                                                  shuffle=True)

    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis=1)

    #plt.style.use('seaborn')
    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, 
             train_scores_mean, 
             label = 'Training error')
    plt.plot(train_sizes, 
             validation_scores_mean, 
             label = 'Validation error')
    plt.ylabel('Mean MAE', 
               fontsize = 14)
    plt.xlabel('Training set size', 
               fontsize = 14)
    plt.title(f'{area}: Learning curves for a {model_title} model', 
              fontsize = 18, 
              y = 1.03)
    plt.legend()
    # plt.ylim(0,40)

## Train vs Val loss
def metrics_graph(model, num_epoch):
    
    r_ep = range(num_epoch)
    train_loss = model.history['loss']
    validation_loss = model.history['val_loss']

    plt.figure(figsize=(15,7))
    
    plt.subplot(1, 2, 1)
    plt.title('Train vs Validation')
    plt.plot(r_ep, train_loss)
    plt.plot(r_ep, validation_loss)
    plt.legend(['train_loss', 'val_loss'])
    plt.xlabel('No. of epochs')
    plt.ylabel('Loss')

## Bins classification for DL
def predict(model, X, y, n_bins):

    y_pred = model.predict(X).flatten()

    percentiles = np.linspace(0, 100, n_bins+1)
    bin_edges = np.percentile(y, percentiles)

    bin_labels = [f"{int(percentiles[i])}-{int(percentiles[i+1])}%" for i in range(len(percentiles) - 1)]

    bin_indices = np.digitize(y_pred, bin_edges, right=True)

    bin_indices = np.clip(bin_indices - 1, 0, len(bin_labels) - 1)

    pred_cat = [bin_labels[i] for i in bin_indices]

    return pred_cat, bin_edges, y_pred    

## Bins calssification for traditional models
def predict_ml(y_pred, y, n_bins):

    percentiles = np.linspace(0, 100, n_bins+1)
    bin_edges = np.percentile(y, percentiles)

    bin_labels = [f"{int(percentiles[i])}-{int(percentiles[i+1])}%" for i in range(len(percentiles) - 1)]

    bin_indices = np.digitize(y_pred, bin_edges, right=True)

    bin_indices = np.clip(bin_indices - 1, 0, len(bin_labels) - 1)

    pred_cat = [bin_labels[i] for i in bin_indices]

    return pred_cat, bin_edges, y_pred 

# ## Multiple loss functions
# def r2_score(y_true, y_pred):
#     ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
#     ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
#     return 1 - (ss_res / (ss_tot + tf.keras.backend.epsilon()))

# def median_absolute_error(y_true, y_pred):
#     error = tf.abs(y_true - y_pred)
#     return tf.numpy_function(np.median, [error], tf.float32)
