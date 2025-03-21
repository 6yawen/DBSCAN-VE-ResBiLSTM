import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np
import seaborn as sns
import os
import glob
import datetime
import math
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import torch.nn.functional as F

####################################  实现每层的残差连接，可以在每层 LSTM 后添加一个残差连接，将前一层的输出和当前层的输入相加。这样，残差将用于每层   ################################################
####################################   训练数据:calculateData4    ##############################################################

# def calculate_average_metrics(test_metrics_dir, fold_num):  # 定义一个计算平均指标的函数，接收测试结果目录和折叠编号作为参数

#     # 列出指定目录中所有以 .xlsx 结尾的文件，存储在 metrics_files 列表中
#     metrics_files = [file for file in os.listdir(test_metrics_dir) if file.endswith('.xlsx')]  #

#     # 初始化多个列表以存储从每个文件中提取的指标（准确率、精度、召回率、F1 分数等）
#     accuracy_list = []

#     field_precision_list = []
#     field_recall_list = []
#     field_f1_list = []

#     road_precision_list = []
#     road_recall_list = []
#     road_f1_list = []

#     precision_macro_list = []
#     recall_macro_list = []
#     f1_macro_list = []

#     for file in metrics_files:  # 遍历每个文件，使用 pandas 读取 Excel 文件，将内容加载到 metrics_df 数据框中
#         metrics_df = pd.read_excel(os.path.join(test_metrics_dir, file))

#         accuracy_list.append(metrics_df['Accuracy'].mean())  # 计算当前文件中“Accuracy”列的平均值，并将其添加到 accuracy_list 列表中。

#         # 计算田间指标
#         # 计算“Field-Precision”、“Field-Recall”和“Field-F1 Score”列的平均值，分别添加到对应的列表中。
#         field_precision_list.append(metrics_df['Field-Precision'].mean())
#         field_recall_list.append(metrics_df['Field-Recall'].mean())
#         field_f1_list.append(metrics_df['Field-F1 Score'].mean())

#         # 计算道路指标
#         # 计算“Road-Precision”、“Road-Recall”和“Road-F1 Score”列的平均值，分别添加到对应的列表中。
#         road_precision_list.append(metrics_df['Road-Precision'].mean())
#         road_recall_list.append(metrics_df['Road-Recall'].mean())
#         road_f1_list.append(metrics_df['Road-F1 Score'].mean())

#         # 计算“Precision_macro”、“Recall_macro”和“F1 Score_macro”列的平均值，分别添加到对应的列表中。
#         precision_macro_list.append(metrics_df['Precision_macro'].mean())
#         recall_macro_list.append(metrics_df['Recall_macro'].mean())
#         f1_macro_list.append(metrics_df['F1 Score_macro'].mean())

#     # 创建一个新的数据框 average_metrics，包含指标名称和对应的平均值。使用 numpy 计算每个列表的平均值。
#     average_metrics = pd.DataFrame({
#         'Metric': ['Accuracy', 'Precision_macro', 'Recall_macro', 'F1 Score_macro', 'Field-Precision', 'Field-Recall',
#                    'Field-F1 Score', 'Road-Precision', 'Road-Recall', 'Road-F1 Score'],
#         'Average Value': [np.mean(accuracy_list), np.mean(precision_macro_list), np.mean(recall_macro_list),
#                           np.mean(f1_macro_list), np.mean(field_precision_list), np.mean(field_recall_list),
#                           np.mean(field_f1_list), np.mean(road_precision_list), np.mean(road_recall_list),
#                           np.mean(road_f1_list)]
#     })
#     # 将“Average Value”列中的所有值四舍五入到小数点后四位
#     average_metrics['Average Value'] = average_metrics['Average Value'].apply(lambda x: round(x, 4))

#     # 将计算得到的平均指标保存到指定目录下的 Excel 文件，文件名包含折叠编号。
#     #timestamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     average_metrics.to_excel(os.path.join(test_metrics_dir, f'average_metrics_{fold_num}.xlsx'), index=False)
#     print("Average metrics calculated and saved.")

def calculate_average_metrics(test_metrics_dir, fold_num):
   
    # 确定当前折叠编号对应的文件名
    target_file = f"evaluation_metrics_{fold_num}.xlsx"
    
    # 检查文件是否存在
    if target_file not in os.listdir(test_metrics_dir):
        print(f"文件 {target_file} 不存在于目录 {test_metrics_dir} 中，无法计算平均指标。")
        return

    # 读取指定文件的内容
    metrics_df = pd.read_excel(os.path.join(test_metrics_dir, target_file))
    
    # 初始化列表存储指标值
    accuracy_list = [metrics_df['Accuracy'].mean()]
    
    field_precision_list = [metrics_df['Field-Precision'].mean()]
    field_recall_list = [metrics_df['Field-Recall'].mean()]
    field_f1_list = [metrics_df['Field-F1 Score'].mean()]
    
    road_precision_list = [metrics_df['Road-Precision'].mean()]
    road_recall_list = [metrics_df['Road-Recall'].mean()]
    road_f1_list = [metrics_df['Road-F1 Score'].mean()]
    
    precision_macro_list = [metrics_df['Precision_macro'].mean()]
    recall_macro_list = [metrics_df['Recall_macro'].mean()]
    f1_macro_list = [metrics_df['F1 Score_macro'].mean()]

    # 创建一个新的数据框存储平均指标
    average_metrics = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision_macro', 'Recall_macro', 'F1 Score_macro', 
                   'Field-Precision', 'Field-Recall', 'Field-F1 Score', 
                   'Road-Precision', 'Road-Recall', 'Road-F1 Score'],
        'Average Value': [np.mean(accuracy_list), np.mean(precision_macro_list), 
                          np.mean(recall_macro_list), np.mean(f1_macro_list), 
                          np.mean(field_precision_list), np.mean(field_recall_list),
                          np.mean(field_f1_list), np.mean(road_precision_list), 
                          np.mean(road_recall_list), np.mean(road_f1_list)]
    })

    # 将“Average Value”列中的所有值四舍五入到小数点后四位
    average_metrics['Average Value'] = average_metrics['Average Value'].apply(lambda x: round(x, 4))

    # 保存结果到文件
    output_file = os.path.join(test_metrics_dir, f'average_metrics_{fold_num}.xlsx')
    average_metrics.to_excel(output_file, index=False)
    print(f"Average metrics for fold {fold_num} calculated and saved to {output_file}.")



# load datasets: each excel as independent image to train
def load_data(file_name):  # 定义一个名为 load_data 的函数，用于从指定文件加载数据
    print(f"Loading data from file {file_name}...")
    df = pd.read_excel(file_name)  # 使用 pandas 从 Excel 文件中读取数据，存储在数据框 df 中

    # feature_columns = ['speed', 'speedDiff', 'distance','acceleration', 'bearing', 'bearingDiff', 'bearingSpeed', 'bearingSpeedDiff','curvature', 'speed_w2_max',  'speed_w2_min', 'speed_w2_var', 'speed_w2_std', 'speed_w2_sum',  'speed_w2_cof','speed_w2_mean', 'bearing_w2_var', 'bearing_w2_std',  'distance_w2_max',  'distance_w2_min', 'distance_w2_var', 'distance_w2_std', 'distance_w2_sum',  'distance_w2_cof','distance_w2_mean', 'acceleration_w2_max', 'acceleration_w2_min',  'acceleration_w2_var', 'acceleration_w2_std', 'acceleration_w2_cof',  'acceleration_w2_mean', 'bearing_speed_w2_max', 'bearing_speed_w2_min', 'bearing_speed_w2_var',  'bearing_speed_w2_std', 'bearing_speed_w2_cof', 'bearing_speed_w2_mean']
    # feature_columns = ['speed', 'speedDiff', 'distance','acceleration', 'bearing', 'bearingDiff', 'bearingSpeed', 'bearingSpeedDiff','curvature']
    # 定义要提取的特征列的列表
    feature_columns = ['distance', 'speed', 'speedDiff', 'acceleration', 'bearing', 'bearingDiff', 'bearingSpeed',
                       'bearingSpeedDiff', 'curvature', 'distance_five', 'distance_ten', 'distribution','angle_std','angle_mean']##,     'mean_length'   ,'radian','mean_length'
    features = df[feature_columns].values  # 从数据框中提取特征列，并将其转换为 NumPy 数组。
    features = features.reshape(-1, 1, len(feature_columns))  # 将特征数组重塑为三维数组，形状为 (样本数, 1, 特征数量)。
    print("Input features shape:", features.shape)  # 输出特征的形状以进行调试
    target = df['type'].values  # 从数据框中提取目标列（标签），并将其转换为 NumPy 数组

    features = torch.tensor(features).float().to('cuda:2')  # 将特征转换为 PyTorch 张量，并移动到 GPU 上，确保数据类型为浮点型。
    target = torch.tensor(target).long().to('cuda:2')  # 将目标标签转换为 PyTorch 张量，并移动到 GPU 上，确保数据类型为长整型。

    print("Data loading completed.")
    return features, target  # 返回特征和目标张量。


class GumbelActivation(nn.Module):  # 定义一个名为 GumbelActivation 的自定义激活函数类，继承自 nn.Module。初始化方法调用父类构造函数
    def __init__(self):
        super(GumbelActivation, self).__init__()

    def forward(self, x):  # 定义前向传播方法，对输入 x 应用 Gumbel 激活函数，返回处理后的结果。
        return torch.exp(-torch.exp(-x))


class VAE(nn.Module):
    def __init__(self, input_dim=14, latent_dim=14):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 28),
            nn.ReLU(),
            nn.Linear(28, 14),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(14, latent_dim)
        self.fc_logvar = nn.Linear(14, latent_dim)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-2, max=2)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(x.size(0), -1)
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class ConvTEModel(nn.Module):
    def __init__(self, num_classes, vae_latent_dim=14, input_dim=14):
        super(ConvTEModel, self).__init__()
        self.vae = VAE(input_dim=input_dim, latent_dim=vae_latent_dim)

        # 设置4层双向LSTM
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size=input_dim if i == 0 else 28, hidden_size=14, num_layers=2,
                    batch_first=True, bidirectional=True) for i in range(2)
        ])
        # 用于将 residual 的维度调整到与 LSTM 输出一致
        self.residual_fc = nn.Linear(input_dim, 28)  # 将 residual 从 12 映射到 24

        self.fc = nn.Linear(28, num_classes)
        self.gumbel_activation = GumbelActivation()

    def forward(self, x):
        # 输入处理：假设输入 x 的形状为 (N, L, C)
        x = torch.mean(x, dim=1)  # x is (N, C)
        x, mu, logvar = self.vae(x)  # 得到潜在变量 z 以及其均值 mu 和对数方差 logvar
        x = x.unsqueeze(1)  # (N, 1, 12)
        x = x.permute(1, 0, 2)  # (1, N, 12)

        # 初始化残差
        residual = self.residual_fc(x)  # residual 的维度为 (1, N, 24)

        # 逐层应用 LSTM 并添加残差连接
        for lstm in self.lstm_layers:
            x, _ = lstm(x)  # x is (L, N, 24)
            x = x + residual  # 每层 BiLSTM 添加残差连接

        # 全局平均池化
        output = torch.mean(x, dim=0)  # x is (N, 24)
        output = self.fc(output)  # 生成 (N, num_classes) 的输出
        output = torch.sigmoid(output)  # 应用 Sigmoid 激活函数
        return output  # 返回最终的模型输出


def test(model, fold_num):  # 定义一个名为 test 的函数，接受模型 model 和折数 fold_num 作为参数

    # 初始化多个列表，用于存储测试过程中的准确率、宏观精确率、召回率和 F1 分数等指标。
    test_accuracy_list = []

    test_precision_macro_list = []
    test_recall_macro_list = []
    test_f1_macro_list = []

    field_precision_list = []
    field_recall_list = []
    field_f1_list = []

    road_precision_list = []
    road_recall_list = []
    road_f1_list = []

    # 定义存储测试结果、图像和热图的文件路径
    test_result_dir = '/home/ubuntu/Data/hyw/filed-road-wheatData/filed-road-wheatData/wheatModelCode/wheatModelCode7--data_pro_all_point/test-(Train6)-6(14)/10/result/'
    test_image_dir = '/home/ubuntu/Data/hyw/filed-road-wheatData/filed-road-wheatData/wheatModelCode/wheatModelCode7--data_pro_all_point/test-(Train6)-6(14)/10/image/'
    test_heatmap_dir = '/home/ubuntu/Data/hyw/filed-road-wheatData/filed-road-wheatData/wheatModelCode/wheatModelCode7--data_pro_all_point/test-(Train6)-6(14)/heatmap/'
    test_metrics_dir='/home/ubuntu/Data/hyw/filed-road-wheatData/filed-road-wheatData/wheatModelCode/wheatModelCode7--data_pro_all_point/test-(Train6)-6(14)/10/metrics/'

    print("Starting model testing...")

    # 使用 glob 模块查找指定路径下所有 .xlsx 文件，存储在 test_file_names 列表中
    test_file_names = glob.glob(
        f"/home/ubuntu/Data/hyw/filed-road-wheatData/filed-road-wheatData/calculateData6(7--data-pro-all-point)/wheat/_10fold/{fold_num}/test/*.xlsx")
    for test_file_name in test_file_names:  # 遍历每个测试文件名。
        model.eval()  # 设置模型为评估模式，以禁用 dropout 和 batch normalization。
        features_test, target_test = load_data(test_file_name)  # 调用 load_data 函数加载测试数据和目标值。
        print(f'features_test: {features_test}')  # 打印测试特征和目标值的内容以进行调试。
        print(f'target_test: {target_test}')
        with torch.no_grad():  # 禁用梯度计算，以减少内存消耗和提高计算速度
            test_outputs = model(features_test)  # 将测试特征输入模型，得到模型输出。
            print(f'test_outputs: {test_outputs}')
            test_predicted = torch.argmax(test_outputs, dim=1).cpu().numpy()  # 获取模型输出的预测类别，使用 argmax 在类别维度上获取最大值的索引。
            # test_predicted = (test_outputs[:, 1] > 0.90).cpu().numpy()
            # test_predicted = test_predicted.astype(int)
            print(f'test_predicted: {test_predicted}')

        target_test_cpu = target_test.cpu().numpy()  # 将目标张量转移到 CPU 并转换为 NumPy 数组，并将数据类型转换为整数
        target_test_cpu = target_test_cpu.astype(int)
        print(f'target_test_cpu: {target_test_cpu}')

        cm = confusion_matrix(target_test_cpu, test_predicted)  # 计算混淆矩阵，以评估模型性能。
        plt.figure(figsize=(8, 6))  # 创建图形，绘制混淆矩阵的热图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Label')  # 设置图形的 x 轴和 y 轴标签。
        plt.ylabel('True Label')

        # 保存混淆矩阵热图并关闭图形。
        #plt.savefig(f'{test_result_dir}{fold_num}/confusionMatrix_{os.path.basename(test_file_name)}.jpg')
        plt.savefig(f'{test_result_dir}{fold_num}/confusionMatrix_{os.path.basename(test_file_name)}.svg',dpi=400,format='svg')
        plt.close()

        # 计算田地类别的精确率、召回率和 F1 分数，并四舍五入到小数点后 5 位。
        field_precision = round(precision_score(target_test_cpu, test_predicted, pos_label=1), 5)
        field_recall = round(recall_score(target_test_cpu, test_predicted, pos_label=1), 5)
        field_f1 = round(f1_score(target_test_cpu, test_predicted, pos_label=1), 5)

        # 将计算得到的田地指标添加到列表中
        field_precision_list.append(field_precision)
        field_recall_list.append(field_recall)
        field_f1_list.append(field_f1)

        # 计算道路类别的精确率、召回率和 F1 分数，并四舍五入到小数点后 5 位。
        road_precision = round(precision_score(target_test_cpu, test_predicted, pos_label=0), 5)
        road_recall = round(recall_score(target_test_cpu, test_predicted, pos_label=0), 5)
        road_f1 = round(f1_score(target_test_cpu, test_predicted, pos_label=0), 5)

        # 将计算得到的道路指标添加到列表中
        road_precision_list.append(road_precision)
        road_recall_list.append(road_recall)
        road_f1_list.append(road_f1)

        test_accuracy = round(accuracy_score(target_test_cpu, test_predicted), 6)  # 计算测试准确率并四舍五入到小数点后 6 位。
        test_accuracy_list.append(test_accuracy)  # 将测试准确率添加到列表中。

        # 计算宏观精确率、召回率和 F1 分数，并四舍五入到小数点后 6 位。
        test_precision_macro = round(precision_score(target_test_cpu, test_predicted, average='macro'), 6)
        test_recall_macro = round(recall_score(target_test_cpu, test_predicted, average='macro'), 6)
        test_f1_macro = round(f1_score(target_test_cpu, test_predicted, average='macro'), 6)

        # 将宏观指标添加到对应的列表中。
        test_precision_macro_list.append(test_precision_macro)
        test_recall_macro_list.append(test_recall_macro)
        test_f1_macro_list.append(test_f1_macro)

        # 打印道路类别的精确率、召回率和 F1 分数。
        print(f'road - Precision: {road_precision}')
        print(f'road - Recall: {road_recall}')
        print(f'road - F1 Score: {road_f1}')

        # 打印田地类别的精确率、召回率和 F1 分数。
        print(f'field - Precision: {field_precision}')
        print(f'field - Recall: {field_recall}')
        print(f'field - F1 Score: {field_f1}')

        # 打印测试准确率和宏观指标。
        print(f'Test Accuracy: {test_accuracy}')
        print(f'Test macro Precision: {test_precision_macro}')
        print(f'Test macro Recall: {test_recall_macro}')
        print(f'Test macro F1 Score: {test_f1_macro}')

        df = pd.read_excel(test_file_name)  # 读取当前测试文件的数据。
        longitude = df['longitude'].values  # 提取经度和纬度数据。
        latitude = df['latitude'].values
        fig, ax = plt.subplots(figsize=(10, 6), dpi=900)  # 创建一个图形和坐标轴，设置图形尺寸和分辨率。

        # triangles = np.where(test_predicted == 0)
        # circles = np.where(test_predicted == 1)
        # ax.scatter(longitude[triangles], latitude[triangles], marker='^', c='black', s=10, label='road')
        # ax.scatter(longitude[circles], latitude[circles], marker='o', c='lightgrey', s=10, label='field')

        colors = np.where(test_predicted == 0, 'blue', 'green')  # 根据预测结果为每个点分配颜色，0 表示道路（蓝色），1 表示田地（绿色）。
        # ax.scatter(longitude, latitude, c=colors, s=5)

        # 在坐标轴上绘制散点图，显示道路和田地的分布。
        ax.scatter(longitude[test_predicted == 0], latitude[test_predicted == 0], c='blue', s=5, label='road')
        ax.scatter(longitude[test_predicted == 1], latitude[test_predicted == 1], c='green', s=5, label='field')

        ax.axis('off')  # 关闭坐标轴显示。
        plt.legend()  # 显示图例。
        plt.savefig(f'{test_image_dir}{fold_num}/{os.path.basename(test_file_name)}.jpg', dpi=900)  # 保存散点图到指定目录。
        plt.close()  # 关闭当前图形。

    eval_df = pd.DataFrame({  # 创建一个 Pandas DataFrame，汇总所有评估指标
        'File': [test_file_name.split('/')[-1].split('.')[0] for test_file_name in test_file_names],
        'Accuracy': test_accuracy_list,
        'Precision_macro': test_precision_macro_list,
        'Recall_macro': test_recall_macro_list,
        'F1 Score_macro': test_f1_macro_list,
        'Field-Precision': field_precision_list,
        'Field-Recall': field_recall_list,
        'Field-F1 Score': field_f1_list,
        'Road-Precision': road_precision_list,
        'Road-Recall': road_recall_list,
        'Road-F1 Score': road_f1_list
    })

    #timestamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_df.to_excel(f'{test_metrics_dir}evaluation_metrics_{fold_num}.xlsx', index=False)  # 将汇总的指标保存到 Excel 文件中。
    calculate_average_metrics(test_metrics_dir, fold_num)  # 调用 calculate_average_metrics 函数计算平均指标。
    print("Model testing completed.")


# 指定折数，加载对应的最佳模型。
fold_num =1
model = torch.load(f'/home/ubuntu/Data/hyw/filed-road-wheatData/filed-road-wheatData/wheatModelCode/wheatModelCode7--data_pro_all_point/bestModel(Train6)-6(14)/best_model10_{fold_num}.pt')
# model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
model = nn.DataParallel(model, device_ids=[2, 3])  # 将模型并行化，以便在多个 GPU 上进行训练。
model = model.to('cuda:2')  # 将模型移动到 GPU 上。

test(model, fold_num)  # 调用 test 函数，传入模型和折数，执行测试
