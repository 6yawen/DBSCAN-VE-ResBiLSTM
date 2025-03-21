# -*- coding: gbk -*-
import os  #用于处理文件路径和文件夹的创建
import pandas as pd  #用于处理数据操作，如读取 Excel 文件和数据处理

input_folder = '/home/ubuntu/Data/zxy/gnss/rawData/paddy'  #原始 .xlsx 数据文件的文件夹路径
output_folder = '/home/ubuntu/Data/zxy/gnss/renameData1/paddy/'  #处理后的文件将被保存到此文件夹

os.makedirs(output_folder, exist_ok=True)  #如果 output_folder 不存在，创建它。exist_ok=True 确保即使文件夹已经存在，也不会报错。

translation_dict = {'时间': 'timestamp',  #将原始 Excel 文件中的中文列名转换为英文名称，以便后续处理
                    '经度': 'longitude',
                    '纬度': 'latitude',
                    '速度': 'speed',
                    '方向': 'bearing',
                    '标记': 'type'}

selected_columns = ['timestamp', 'longitude', 'latitude', 'speed', 'bearing', 'type']  #选择列：指定希望保留的列名，用于输出
date_format = "%Y/%m/%d %H:%M:%S"  #定义日期格式，用于格式化 timestamp 列中的时间数据

#遍历输入文件夹中的所有文件：循环检查 input_folder 中的文件，筛选出以 .xlsx 结尾的 Excel 文件。
for file_name in os.listdir(input_folder):
    if file_name.endswith('.xlsx'):
        input_file_path = os.path.join(input_folder, file_name)  #输入文件的完整路径
        output_file_path = os.path.join(output_folder, file_name)  #输出文件的完整路径

        df = pd.read_excel(input_file_path)  #使用 pandas 读取 Excel 文件内容，并存储在 df（DataFrame）对象中。
        df = df.rename(columns=translation_dict)  #将 DataFrame 中的列名根据 translation_dict 进行重命名
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime(date_format)#将 timestamp 列转换为标准的日期时间格式，并格式化为指定的 date_format
        df.loc[df['bearing'] == 360, 'bearing'] = 0 #处理方向列中的特殊值：将 bearing（方向）列中值为 360 的数据替换为 0
        df = df.sort_values(by='timestamp')  #按时间戳排序：根据 timestamp 列对数据进行升序排序
        
        #delete time repeat point
        time_delete = []  #time_delete：记录重复时间点的索引
        all_time_point = []  #all_time_point：存储所有非重复的时间点。
        timestamp = df['timestamp']
        for j in range(len(timestamp)):  ##遍历 timestamp 列，将重复的时间点标记为删除对象
            if timestamp[j] in all_time_point:
                time_delete.append(j)
            else:
                all_time_point.append(timestamp[j])

         ##根据 time_delete 中的索引删除重复的行，并重置 DataFrame 的索引
        df = df.drop(time_delete)
        df = df.reset_index(drop=True)
       
        #delete space repeat point删除重复的空间点
        longitude = df['longitude']
        latitude = df['latitude']
        speed = df['speed']      
        space_delete = []  #记录重复空间点的索引
        all_space_point = []  #存储所有非重复的空间点（包括经度、纬度和速度）
        for k in range(len(speed)): #遍历经度、纬度、速度列，将重复的空间点标记为删除对象
            point = []
            point = [longitude[k], latitude[k], speed[k]]
            if point in all_space_point:
                space_delete.append(k)
            else:
                all_space_point.append(point)

         #根据 space_delete 中的索引删除重复的空间点，并重置 DataFrame 的索引。
        df = df.drop(space_delete)
        df = df.reset_index(drop=True)

         #将处理后的数据（仅保留 selected_columns 中的列）保存为新的 Excel 文件，不包括行索引。
        df[selected_columns].to_excel(output_file_path, index=False)

