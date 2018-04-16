import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pandas import DataFrame, Series, concat, to_numeric
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from scipy.optimize import curve_fit

def remove_outliers(df, x_col, y_col):
    df = df.apply(to_numeric).dropna()

    df_local = DataFrame.from_dict({
        x_col: df.filter(regex=x_col).mean(axis=1),
        y_col: df[y_col]
    }, dtype=np.float64)
    
    cluster_col = 'Cluster'    

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    norm_x = x_scaler.fit_transform(df_local[x_col].values.reshape(-1, 1))
    norm_y = y_scaler.fit_transform(df_local[y_col].values.reshape(-1, 1))

    norm_df = DataFrame.from_dict({
        x_col: Series(norm_x[:, 0]), 
        y_col: Series(norm_y[:, 0])
    })

    clusters = create_clusters(norm_df)
    df_local[cluster_col] = clusters
    norm_df[cluster_col] = clusters

    largest_cluster = get_largest_cluster_points(norm_df, cluster_col)
    x_lower, y_lower, x_upper, y_upper = get_cluster_bounds(largest_cluster, x_col, y_col, x_scaler, y_scaler)
    bounds_status = map(lambda pt: respects_bounds(pt, x_lower, y_lower, x_upper, y_upper), zip(df_local[x_col].values, df_local[y_col].values))

    return df[list(bounds_status)]

def respects_bounds(point, x_lower, y_lower, x_upper, y_upper):
    respects_lower_bounds = True
    respects_upper_bounds = True

    if point[0] >= 20 and point[1] >= 40:
        return True
    
    for line_point in zip(x_lower, y_lower):
        if line_point[0] > point[0]:
            break
        if line_point[1] > point[1]:
            respects_lower_bounds = False
            
    for line_point in zip(x_upper, y_upper):
        if line_point[1] > point[1]:
            break
        if line_point[0] > point[0]:
            respects_lower_bounds = False
    
    return respects_lower_bounds and respects_lower_bounds

def sigmoid(x, x0, k):
    y = 1 / (1 + np.exp(-k*(x-x0)))
    return y

def create_clusters(df, cluster_eps=0.03, min_samples=30):
    dbscan = DBSCAN(eps=cluster_eps, min_samples=min_samples)
    return dbscan.fit_predict(df)

# Retrieves all points beloning to the largest cluster, not considering the null cluster
def get_largest_cluster_points(df, cluster_col):
    sizes = df.groupby(cluster_col).size()
    df = df[df[cluster_col] != -1]
    largest = sizes[sizes == max(sizes)].index[0]
    return df[df[cluster_col] == largest]

def get_cluster_bounds(df, x_col, y_col, x_scaler, y_scaler):
    min_pts = []
    max_pts = []

    interval_min = 0
    interval_offset = 0.001

    while interval_min + interval_offset < 1:
        interval_max = interval_min + interval_offset
        points_in_interval = df[(df[x_col] > interval_min) & (df[x_col] < interval_max)]
        
        min_y = points_in_interval[points_in_interval[y_col] == points_in_interval[y_col].min()]
        max_y = points_in_interval[points_in_interval[y_col] == points_in_interval[y_col].max()]
        
        if len(min_y[x_col].values) == 0:
            interval_min += interval_offset
            continue
        min_pts.append([min_y[x_col].values[0], min_y[y_col].values[0]])
        max_pts.append([max_y[x_col].values[0], max_y[y_col].values[0]])
        interval_min += interval_offset

    min_df = DataFrame(min_pts, columns=[x_col, y_col])
    max_df = DataFrame(max_pts, columns=[x_col, y_col])

    popt_lower, pcov_lower = curve_fit(sigmoid, min_df[x_col], min_df[y_col]) 
    popt_upper, pcov_upper = curve_fit(sigmoid, max_df[x_col], max_df[y_col])

    x_lower = x_scaler.inverse_transform(min_df[x_col].values.reshape(-1,1))
    x_upper = x_scaler.inverse_transform(max_df[x_col].values.reshape(-1,1))

    y_lower = y_scaler.inverse_transform(sigmoid(min_df[x_col].values.reshape(-1,1), *popt_lower)) - 4
    y_upper = y_scaler.inverse_transform(sigmoid(max_df[x_col].values.reshape(-1,1), *popt_upper)) + 4

    return x_lower, y_lower, x_upper, y_upper