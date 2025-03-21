# -*- coding: gbk -*-
import os  
import pandas as pd  

input_folder = '/home/ubuntu/Data/zxy/gnss/rawData/paddy'  
output_folder = '/home/ubuntu/Data/zxy/gnss/renameData1/paddy/'  

os.makedirs(output_folder, exist_ok=True)  

translation_dict = {'时间': 'timestamp', 
                    '经度': 'longitude',
                    '纬度': 'latitude',
                    '速度': 'speed',
                    '方向': 'bearing',
                    '标记': 'type'}

selected_columns = ['timestamp', 'longitude', 'latitude', 'speed', 'bearing', 'type']  
date_format = "%Y/%m/%d %H:%M:%S"  #定义日期格式，用于格式化 timestamp 列中的时间数据

#遍历输入文件夹中的所有文件：循环检查 input_folder 中的文件，筛选出以 .xlsx 结尾的 Excel 文件。
for file_name in os.listdir(input_folder):
    if file_name.endswith('.xlsx'):
        input_file_path = os.path.join(input_folder, file_name)  
        output_file_path = os.path.join(output_folder, file_name)  

        df = pd.read_excel(input_file_path) 
        df = df.rename(columns=translation_dict) 
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime(date_format)
        df.loc[df['bearing'] == 360, 'bearing'] = 0 
        df = df.sort_values(by='timestamp')  
        
        #delete time repeat point
        time_delete = []  
        all_time_point = []  
        timestamp = df['timestamp']
        for j in range(len(timestamp)):  
            if timestamp[j] in all_time_point:
                time_delete.append(j)
            else:
                all_time_point.append(timestamp[j])

       
        df = df.drop(time_delete)
        df = df.reset_index(drop=True)
       
       
        longitude = df['longitude']
        latitude = df['latitude']
        speed = df['speed']      
        space_delete = []  
        all_space_point = []  
        for k in range(len(speed)): 
            point = []
            point = [longitude[k], latitude[k], speed[k]]
            if point in all_space_point:
                space_delete.append(k)
            else:
                all_space_point.append(point)

         
        df = df.drop(space_delete)
        df = df.reset_index(drop=True)

        
        df[selected_columns].to_excel(output_file_path, index=False)

