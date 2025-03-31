import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import NearestNeighbors
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.interpolate import splprep, splev
import seaborn as sns
from statsmodels.tsa.stattools import acf # 用于计算自相关

from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return 6371 * c  # 地球半径，单位公里


# 计算地理距离，使用Haversine公式
def compute_distance(lat1, lon1, lat2, lon2):
    R = 6371  # 地球半径，单位为千米
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c  # 返回距离，单位为千米
    return distance


def haversine_distances(coords1, coords2):
    num_points1 = len(coords1)
    num_points2 = len(coords2)
    distances = np.zeros((num_points1, num_points2))

    for i in range(num_points1):
        for j in range(num_points2):
            lat1, lon1 = coords1[i]
            lat2, lon2 = coords2[j]
            dlat = lat2 - lat1
            dlon = lon2 - lon1

            # Haversine公式
            a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            distances[i, j] = 6371 * c  # 地球半径 6371 公里

    return distances


def calculate_metrics(data):
    # 计算经纬度的均值
    mean_latitude = data['latitude'].mean()
    mean_longitude = data['longitude'].mean()

    # 计算经纬度的方差
    variance_latitude = data['latitude'].var()
    variance_longitude = data['longitude'].var()

    # 计算经纬度的标准差
    stddev_latitude = data['latitude'].std()
    stddev_longitude = data['longitude'].std()

    # 计算经纬度的极差
    range_latitude = data['latitude'].max() - data['latitude'].min()
    range_longitude = data['longitude'].max() - data['longitude'].min()

    # 综合指标
    overall_mean = (mean_latitude + mean_longitude) / 2  # 经纬度均值的平均
    overall_variance = (variance_latitude + variance_longitude) / 2  # 经纬度方差的平均
    overall_stddev = (stddev_latitude + stddev_longitude) / 2  # 经纬度标准差的平均
    overall_range = (range_latitude + range_longitude) / 2  # 经纬度极差的平均

    return overall_mean, overall_variance, overall_stddev, overall_range



def is_direction_consistent(bearing1, bearing2, max_angle_diff=40):
    return abs(bearing1 - bearing2) <= max_angle_diff or abs(bearing1 - bearing2) >= 360 - max_angle_diff


savgol_window = 11  
savgol_polyorder = 2  

def smooth_trajectory(data):

    if len(data) < savgol_window:
        print("点数不足，跳过平滑处理。")
        return data

    data['latitude'] = savgol_filter(data['latitude'], savgol_window, savgol_polyorder)
    data['longitude'] = savgol_filter(data['longitude'], savgol_window, savgol_polyorder)
    return data



def calculate_average_autocorrelation(data, max_lag=25):
    autocorr_values = acf(data, nlags=max_lag, fft=False)
    # 去除滞后0的自相关值
    autocorr_values = autocorr_values[1:max_lag+1]
    average_autocorr = np.mean(autocorr_values)
    return average_autocorr



# 文件路径
folder_path = '/home/ubuntu/Data/renameData/wheat/'
log_file_path = '/home/ubuntu/Data/renameData/interpolation_log.txt'
log_file_path_1 = '/home/ubuntu/Data/renameData/interpolation_1_log.txt'


# DBSCAN参数
eps_km=0.027   
min_samples =3 


distance_threshold = 0.027      
interpolation_interval = 0.005  

max_bearing_diff = 30  # 最大方向差，单位度

# 清空或创建日志文件
with open(log_file_path, 'w') as log_file:
    log_file.write("插值处理日志\n\n")

with open(log_file_path_1, 'w') as log_file:
    log_file.write("平滑前后的数据指标\n\n")

original_means = []  #均值
original_variances = []  #方差
original_stddev = []  #标准差
original_range = []  #极差

smoothed_means = []
smoothed_variances = []
smoothed_stddev = []
smoothed_range = []

original_autocorr = []
smoothed_autocorr = []


# 遍历文件夹中的每个 Excel 文件
for filename in os.listdir(folder_path):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(folder_path, filename)

        # 读取数据
        data = pd.read_excel(file_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'])

       
        orig_mean, orig_variance, orig_stddev, orig_range = calculate_metrics(data)
        original_means.append(orig_mean)
        original_variances.append(orig_variance)
        original_stddev.append(orig_stddev)
        original_range.append(orig_range)  # 极差


        
        road_data = data[data['type'] == 0].reset_index(drop=True)  # 道路点

        field_data = data[data['type'] == 1].reset_index(drop=True)  # 田间点  

        if len(road_data) < min_samples:
            print(f"文件 '{filename}' 的道路点不足，跳过处理。")
            continue

        smooth_field_data = smooth_trajectory(field_data)  
        
        smooth_road_data = smooth_trajectory(road_data)

        smooth_data = pd.concat([smooth_road_data, smooth_field_data], ignore_index=True).sort_values(by='timestamp').reset_index(drop=True)

        # 计算平滑后的均值和方差
        smoo_mean, smoo_variance, smoo_stddev, smoo_range = calculate_metrics(smooth_data)

        smoothed_means.append(smoo_mean)
        smoothed_variances.append(smoo_variance)
        smoothed_stddev.append(smoo_stddev)
        smoothed_range.append(smoo_range)

        print(f"{filename}平滑结束")


        
        coordinates = smooth_road_data[['latitude', 'longitude']].apply(lambda x: x.apply(radians)).to_numpy()
        distance_matrix = haversine_distances(coordinates, coordinates)  # 结果是公里
        # DBSCAN聚类，使用自定义的距离矩阵
        db = DBSCAN(eps=eps_km, min_samples=min_samples, metric="precomputed").fit(distance_matrix)
        smooth_road_data['cluster'] = db.labels_

        noise_points = smooth_road_data[smooth_road_data['cluster'] == -1]
        print(f"噪声点数量：{len(noise_points)}")

        # 提取簇标签
        clusters = smooth_road_data['cluster']

        # 获取聚类数量（排除噪声点）
        unique_clusters = set(clusters)
        num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)


        # 输出聚类数量和噪声点数量
        print(f"DBSCAN 聚类结果: 共聚成 {num_clusters} 个类（簇）。")
        if -1 in unique_clusters:
            print(f"此外，有噪声点：{len(smooth_road_data[smooth_road_data['cluster'] == -1])} 个。")
        else:
            print("没有噪声点。")

       
        for cluster_label in unique_clusters:
            if cluster_label == -1:
                print(f"噪声点: {len(smooth_road_data[smooth_road_data['cluster'] == -1])} 个")
            else:
                cluster_size = len(smooth_road_data[smooth_road_data['cluster'] == cluster_label])
                print(f"簇 {cluster_label}: 包含 {cluster_size} 个点")


        cluster_counts = smooth_road_data['cluster'].value_counts()

        colormap = matplotlib.colormaps['tab20']  # 使用新的方式获取colormap
       
        plt.figure(figsize=(10, 8))

        for cluster_label in unique_clusters:
            cluster_points = road_data[road_data['cluster'] == cluster_label]

            if cluster_label == -1:
               
                plt.scatter(cluster_points['longitude'], cluster_points['latitude'], c='black', label='Noise', s=10,
                            alpha=0.6)
            else:
                color = colormap(cluster_label % num_clusters) 
                plt.scatter(cluster_points['longitude'], cluster_points['latitude'], c=[color],
                            label=f'Cluster {cluster_label}', s=10, alpha=0.8)

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f"聚类结果 - 文件: {filename}")
        plt.legend(loc='best')
        plt.savefig(os.path.join(folder_path, f"{filename}_clusters.png"))
        plt.close()

        new_points = []
        for cluster_label in set(smooth_road_data['cluster']):
            if cluster_label == -1:
                continue  

            cluster_points = smooth_road_data[smooth_road_data['cluster'] == cluster_label].sort_values(by='timestamp').reset_index(drop=True)

            for i in range(len(cluster_points) - 1):
                current = cluster_points.iloc[i]
                next = cluster_points.iloc[i + 1]
                distance = haversine(current['latitude'], current['longitude'], next['latitude'], next['longitude'])

                if distance > distance_threshold:
                    continue
                if not is_direction_consistent(current['bearing'], next['bearing'], max_angle_diff=max_bearing_diff):
                    continue

               
                num_points = int(distance / interpolation_interval)
                latitudes = np.linspace(current['latitude'], next['latitude'], num_points + 2)[1:-1]
                longitudes = np.linspace(current['longitude'], next['longitude'], num_points + 2)[1:-1]
                speeds = np.linspace(current['speed'], next['speed'], num_points + 2)[1:-1]
                bearings = np.linspace(current['bearing'], next['bearing'], num_points + 2)[1:-1]
                time_deltas = np.linspace(0, 1, num_points + 2)[1:-1]
                timestamps = [
                    current['timestamp'] + (next['timestamp'] - current['timestamp']) * delta
                    for delta in time_deltas
                ]

                for lat, lon, spd, brg, ts in zip(latitudes, longitudes, speeds, bearings, timestamps):
                    new_points.append({
                        'latitude': lat,
                        'longitude': lon,
                        'speed': spd,
                        'bearing': brg,
                        'timestamp': ts,
                        'type': 0
                    })

        
        interpolated_data = pd.DataFrame(new_points)
        combined_data = pd.concat([smooth_data, interpolated_data], ignore_index=True).sort_values(by='timestamp').reset_index(drop=True)

        combined_data['timestamp'] = combined_data['timestamp'].dt.strftime('%Y/%m/%d %H:%M:%S')
        
        combined_data.to_excel(file_path, index=False)


       
        log_message = (
            f"文件 '{filename}' 处理完成：\n"
            f" - 原始道路点数量：{len(smooth_road_data)}\n"
            f" - 插值生成的道路点数量：{len(interpolated_data)}\n"
            f" - 插值后总轨迹点数量：{len(combined_data)}\n\n"
            f" - 此外，有噪声点：{len(smooth_road_data[smooth_road_data['cluster'] == -1])} 个。\n"
            f" - DBSCAN 聚类结果: 共聚成 {num_clusters} 个类（簇）。\n"

        )
        print(log_message)
        with open(log_file_path, 'a') as log_file:
            log_file.write(log_message)


if len(original_means) > 0 and len(smoothed_means) > 0 and len(original_variances)>0 and len(smoothed_variances)>0:
    print("平滑前后的数据指标")

    log_message = (
        f"-平滑前均值列表长度：{len(original_means)}\n"
        f"-Original Means Mean: {np.mean(original_means)}\n"
        f"-Original Variances Mean: {np.mean(original_variances)}\n"
        f"-Original Standard Deviation Mean: {np.mean(original_stddev)}\n"
        f"-Original Range Mean: {np.mean(original_range)}\n"
        f"-平滑后均值列表长度：{len(smoothed_means)}\n"
        f"-Smoothed Means Mean: {np.mean(smoothed_means)}\n"
        f"-Smoothed Variances Mean: {np.mean(smoothed_variances)}\n"
        f"-Smoothed Standard Deviation Mean: {np.mean(smoothed_stddev)}\n"
        f"-Smoothed Range Mean: {np.mean(smoothed_range)}\n"

        f"Original Autocorrelation Mean: {np.mean(original_autocorr):.4f}\n"
        f"Smoothed Autocorrelation Mean: {np.mean(smoothed_autocorr):.4f}\n"

            )
    with open(log_file_path_1, 'a') as log_file:
        log_file.write(log_message)

else:
    print("均值或方差数据不足，请检查数据处理逻辑！")
