import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import math
import glob
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import seaborn as sns
import random
import torch
import torch.nn as nn

####################################  实现每层的残差连接，可以在每层 LSTM 后添加一个残差连接，将前一层的输出和当前层的输入相加。这样，残差将用于每层   ################################################
####################################   训练数据:calculateData4    ##############################################################
# load datasets: each excel as independent image to train
def load_data(file_name):  # 函数从指定的 Excel 文件中加载数据
    print(f"Loading data from file {file_name}...")  # file_name：传入的 Excel 文件的路径。
    df = pd.read_excel(file_name)  # 使用 pandas 读取 Excel 文件，并将其存储在 df（DataFrame）中。

    # feature_columns = ['speed', 'speedDiff', 'distance','acceleration', 'bearing', 'bearingDiff', 'bearingSpeed', 'bearingSpeedDiff','curvature', 'speed_w2_max',  'speed_w2_min', 'speed_w2_var', 'speed_w2_std', 'speed_w2_sum',  'speed_w2_cof','speed_w2_mean', 'bearing_w2_var', 'bearing_w2_std',  'distance_w2_max',  'distance_w2_min', 'distance_w2_var', 'distance_w2_std', 'distance_w2_sum',  'distance_w2_cof','distance_w2_mean', 'acceleration_w2_max', 'acceleration_w2_min',  'acceleration_w2_var', 'acceleration_w2_std', 'acceleration_w2_cof',  'acceleration_w2_mean', 'bearing_speed_w2_max', 'bearing_speed_w2_min', 'bearing_speed_w2_var',  'bearing_speed_w2_std', 'bearing_speed_w2_cof', 'bearing_speed_w2_mean']
    # feature_columns = ['speed', 'speedDiff', 'distance','acceleration', 'bearing', 'bearingDiff', 'bearingSpeed', 'bearingSpeedDiff','curvature']

    # feature_columns：定义一个包含要用作输入特征的列名的列表。
    feature_columns = ['distance', 'speed', 'speedDiff', 'acceleration', 'bearing', 'bearingDiff', 'bearingSpeed',
                       'bearingSpeedDiff', 'curvature', 'distance_five', 'distance_ten', 'distribution','angle_std','angle_mean'] #少了，6-1  'mean_length' 'radian'
    features = df[feature_columns].values  # 从 DataFrame 中提取这些列作为特征数据； .values：将这些列的值转换为 NumPy 数组，便于后续操作

    # 将特征数据的形状调整为 PyTorch 模型训练所需的 3D 张量。
    # 将特征的形状调整为 (样本数量, 1, 特征数量)，其中 -1 表示自动计算样本数量，1 是通道维度，用于将每个样本视为一个一维图像。形状调整的目的是让数据适配卷积神经网络的输入格式。
    features = features.reshape(-1, 1, len(feature_columns))  # 将 features 数组重新调整为 (N, 1, F) 的形状；N 是样本的数量（-1 表示自动推断）；
    # 1 表示每个样本有 1 个通道（可以类比为图像的单通道） ；  F 是特征数量，即列的数量（由 len(feature_columns) 获得）。
    print("Input features shape:", features.shape)  # 打印调整后输入特征的形状，用于调试和检查数据是否符合预期。
    target = df['type'].values  # 从 DataFrame 中提取 type 列（目标列，表示该点是田地还是道路的类型），并将其转换为 NumPy 数组。

    # 将 NumPy 数组转换为 PyTorch 张量，并准备好在 GPU 上训练。
    ##torch.tensor(features)：将 features 数组转换为 PyTorch 张量。
    features = torch.tensor(features).float().to('cuda:6')  # .float()：将特征数据转换为浮点类型，通常用于神经网络中的输入；.to('cuda')：将数据移动到 GPU（如果可用），以加快训练速度。
    target = torch.tensor(target).long().to('cuda:6')  # 同样，target 也转换为长整型张量（long()），因为 PyTorch 中的分类任务标签通常使用长整型

    print("Data loading completed.")
    return features, target  # 返回两个张量（features 和 target），分别表示输入特征和目标标签，这两个值将在后续模型训练中使用


class GumbelActivation(nn.Module):  # 定义一个自定义激活函数 GumbelActivation，继承自 nn.Module。这个类将使用 Gumbel 分布的激活函数
    def __init__(self):
        super(GumbelActivation, self).__init__()  # super 调用：调用父类的 __init__ 初始化方法，确保正确初始化 nn.Module 类。

    def forward(self, x):  # 定义 forward 方法，表示在前向传播时应用的操作
        return torch.exp(-torch.exp(-x))  # 这是 Gumbel 分布的累积分布函数（CDF）。它将输入 x 转换为符合 Gumbel 分布的值，用于激活非线性处理。


class VAE(nn.Module):
    def __init__(self, input_dim=14, latent_dim=14):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 28),
            nn.ReLU(),
            nn.Linear(28, 14),
        #   nn.Linear(28, 24),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(14, latent_dim)
        self.fc_logvar = nn.Linear(14, latent_dim)
        #self.fc_mu = nn.Linear(24, latent_dim)
        #self.fc_logvar = nn.Linear(24, latent_dim)

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

        #self.lstm_layers = nn.ModuleList([
        #   nn.LSTM(input_size=input_dim if i == 0 else 28, hidden_size=24, num_layers=2,
        #            batch_first=True, bidirectional=True) for i in range(2)
       #])
        # 用于将 residual 的维度调整到与 LSTM 输出一致
        self.residual_fc = nn.Linear(input_dim, 28)  # 将 residual 从 12 映射到 24

        #self.residual_fc = nn.Linear(input_dim, 48)  # 将 residual 从 12 映射到 24

        self.fc = nn.Linear(28, num_classes)

        #self.fc = nn.Linear(48, num_classes)
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


# FocalLossWithRegularization 类的作用是定义一种结合焦点损失和L2正则化的损失函数，主要用于处理类别不平衡问题。
class FocalLossWithRegularization(nn.Module):
    # alpha：焦点损失的类权重，控制正负样本的影响。;gamma：聚焦参数，调整易分类样本的权重。;reduction：损失的归约方式（如求均值或总和）。;regularization_coeff：L2正则化系数，防止过拟合
    def __init__(self, alpha, gamma, reduction, regularization_coeff=0.00001):
        super(FocalLossWithRegularization, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.regularization_coeff = regularization_coeff

    def forward(self, inputs, targets):  # 计算每个样本的交叉熵损失，inputs 为模型的输出，targets 为真实标签
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 计算焦点损失
        pt = torch.exp(-ce_loss)  # pt：计算样本预测的概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # 通过焦点损失公式，结合 alpha 和 gamma 计算最终的焦点损失。

        # 根据 reduction 的值，对损失进行均值或总和的归约
        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss)

        # L2 regularization计算模型所有参数的L2正则化项，即所有参数的平方和
        regularization_term = 0.0
        for param in self.parameters():
            regularization_term += torch.sum(param.pow(2))

        # 将焦点损失和正则化项相加，返回最终损失
        total_loss = focal_loss + self.regularization_coeff * regularization_term  # Combine Focal Loss and Regularization

        return total_loss  # 返回最终损失


def train(model, num_epochs, batch_size, patience=10):  # 定义一个训练模型的函数，接收模型、训练轮数、批量大小和早停耐心参数。
    # optimizer = optim.Adam(model.parameters(), lr=0.00001) ok
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.00001)  # 使用 AdamW 优化器来优化模型参数，设置学习率和权重衰减。
    criterion = FocalLossWithRegularization(alpha=0.25, gamma=2, reduction='mean')  # 定义带有正则化的焦点损失函数

    # 初始化最优模型状态和各类历史记录列表（训练损失、验证精度等）
    best_model_state = None
    train_loss_history = []
    val_accuracy_history = []

    val_precision_macro_history = []
    val_recall_macro_history = []
    val_f1_macro_history = []

    train_loss_avg_history = []
    val_loss_avg_history = []

    # 输出训练开始信息，使用 glob 加载指定路径下的训练数据文件，并将特征和目标数据存入 train_data 列表。
    print("Starting model training...")
    train_data = []
    train_file_names = glob.glob(
        f"/home/ubuntu/Data/hyw/filed-road-wheatData/filed-road-wheatData/calculateData6(7--data-pro-all-point)/wheat/_10fold/{fold_num}/train/*.xlsx")
    for train_file_name in train_file_names:
        features_train, target_train = load_data(train_file_name)
        train_data.append((features_train, target_train))

    # 初始化早停计数器和最佳混合结果
    early_stopping_counter = 0
    best_mix_result = 0

    for epoch in range(num_epochs):  # 开始训练的主循环，遍历每个 epoch，将模型设为训练模式并初始化总损失
        model.train()  # 将模型设为训练模式
        all_file_loss = 0.0  # 初始化总损失
        for train_file_index, (features_train, target_train) in enumerate(train_data, 1):  # 遍历每个训练文件，初始化当前文件损失
            current_file_loss = 0.0  # 初始化当前文件损失
            batch_size = features_train.shape[0]

            for i in range(0, len(features_train), batch_size):  # 按批量大小遍历训练数据，将数据移至 GPU 并转换为适当的数据类型
                batch_features = features_train.clone().detach().to('cuda:6').float()
                batch_target = target_train.clone().detach().to('cuda:6').long()

                # 清零梯度，执行前向传播得到输出，计算损失，反向传播梯度并更新模型参数。
                optimizer.zero_grad()
                output = model(batch_features)

                loss = criterion(output, batch_target)  # setting
                loss.backward()
                optimizer.step()

                current_file_loss += loss.item()  # 累加当前文件的损失并计算平均损失
            current_file_average_loss = current_file_loss

            # 输出每个文件和总的训练损失
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], File: {train_file_index}-Training Loss: {current_file_average_loss}')
            all_file_loss += current_file_average_loss
        all_file_average_loss = all_file_loss / len(train_data)
        print(f'Epoch [{epoch + 1}/{num_epochs}], all_file average training Loss: {all_file_average_loss}')

        if (epoch + 1) % 10 == 0:  # 每10个 epoch 进行一次验证，记录验证集的损失和各类指标

            train_loss_avg_history.append(all_file_average_loss)

            val_loss_list = []
            val_accuracy_list = []
            val_precision_macro_list = []
            val_recall_macro_list = []
            val_f1_macro_list = []

            field_precision_list = []
            field_recall_list = []
            field_f1_list = []

            road_precision_list = []
            road_recall_list = []
            road_f1_list = []

            # 加载验证数据文件，设定模型为评估模式
            val_file_names = glob.glob(
                f"/home/ubuntu/Data/hyw/filed-road-wheatData/filed-road-wheatData/calculateData6(7--data-pro-all-point)/wheat/_10fold/{fold_num}/valid/*.xlsx")
            for val_file_name in val_file_names:
                model.eval()  # 设定模型为评估模式
                features_val, target_val = load_data(val_file_name)
                print(f'features_val: {features_val}')
                print(f'target_val: {target_val}')

                with torch.no_grad():  # 在不计算梯度的上下文中进行前向传播，计算预测结果，并进行评估指标（精度、召回率、F1 分数等）的计算。
                    val_outputs = model(features_val)
                    print(f'val_outputs: {val_outputs}')
                    val_predicted = torch.argmax(val_outputs, dim=1).cpu().numpy()
                    print(f'val_predicted: {val_predicted}')
                target_val_cpu = target_val.cpu().numpy()
                target_val_cpu = target_val_cpu.astype(int)
                print(f'target_val_cpu: {target_val_cpu}')

                field_precision = round(precision_score(target_val_cpu, val_predicted, pos_label=1), 5)
                field_recall = round(recall_score(target_val_cpu, val_predicted, pos_label=1), 5)
                field_f1 = round(f1_score(target_val_cpu, val_predicted, pos_label=1), 5)

                field_precision_list.append(field_precision)
                field_recall_list.append(field_recall)
                field_f1_list.append(field_f1)

                road_precision = round(precision_score(target_val_cpu, val_predicted, pos_label=0), 5)
                road_recall = round(recall_score(target_val_cpu, val_predicted, pos_label=0), 5)
                road_f1 = round(f1_score(target_val_cpu, val_predicted, pos_label=0), 5)

                road_precision_list.append(road_precision)
                road_recall_list.append(road_recall)
                road_f1_list.append(road_f1)

                val_loss = criterion(val_outputs, target_val).item()  # setting
                val_accuracy = round(accuracy_score(target_val_cpu, val_predicted), 6)

                val_precision_macro = round(precision_score(target_val_cpu, val_predicted, average='macro'), 6)
                val_recall_macro = round(recall_score(target_val_cpu, val_predicted, average='macro'), 6)
                val_f1_macro = round(f1_score(target_val_cpu, val_predicted, average='macro'), 6)

                val_loss_list.append(val_loss)
                val_accuracy_list.append(val_accuracy)

                val_precision_macro_list.append(val_precision_macro)
                val_recall_macro_list.append(val_recall_macro)
                val_f1_macro_list.append(val_f1_macro)

                print(f'road Precision: {road_precision}')
                print(f'road Recall: {road_recall}')
                print(f'road F1 score: {road_f1}')

                print(f'field Precision: {field_precision}')
                print(f'field Recall: {field_recall}')
                print(f'field F1 score: {field_f1}')

                print(f'Validation Loss: {val_loss}')
                print(f'Validation Accuracy: {val_accuracy}')
                print(f'Validation macro Precision: {val_precision_macro}')
                print(f'Validation macro Recall: {val_recall_macro}')
                print(f'Validation macro F1 score: {val_f1_macro}')

                # 计算平均验证损失和其他指标，并输出。
            val_loss_avg = sum(val_loss_list) / len(val_loss_list)
            val_accuracy_avg = round(sum(val_accuracy_list) / len(val_accuracy_list), 6)
            val_precision_macro_avg = round(sum(val_precision_macro_list) / len(val_precision_macro_list), 7)
            val_recall_macro_avg = round(sum(val_recall_macro_list) / len(val_recall_macro_list), 7)
            val_f1_macro_avg = round(sum(val_f1_macro_list) / len(val_f1_macro_list), 7)

            print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss_avg}')
            print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {val_accuracy_avg}')
            print(f'Epoch [{epoch + 1}/{num_epochs}], Validation macro Precision: {val_precision_macro_avg}')
            print(f'Epoch [{epoch + 1}/{num_epochs}], Validation macro Recall: {val_recall_macro_avg}')
            print(f'Epoch [{epoch + 1}/{num_epochs}], Validation macro F1 score: {val_f1_macro_avg}')

            val_accuracy_history.append(val_accuracy_avg)
            val_loss_avg_history.append(val_loss_avg)

            val_precision_macro_history.append(val_precision_macro_avg)
            val_recall_macro_history.append(val_recall_macro_avg)
            val_f1_macro_history.append(val_f1_macro_avg)

            # 根据验证结果更新最佳模型和早停计数器。如果达到早停耐心值，停止训练
            if 0.2 * val_precision_macro_avg + 0.2 * val_recall_macro_avg + 0.3 * val_f1_macro_avg + 0.3 * val_accuracy_avg > best_mix_result:
                best_mix_result = 0.2 * val_precision_macro_avg + 0.2 * val_recall_macro_avg + 0.3 * val_f1_macro_avg + 0.3 * val_accuracy_avg
                best_model_state = model.state_dict()
                early_stopping_counter = 0
                torch.save(model.module,
                           f'/home/ubuntu/Data/hyw/filed-road-wheatData/filed-road-wheatData/wheatModelCode/wheatModelCode7--data_pro_all_point/bestModel(Train6)-6(14)/best_model10_{fold_num}.pt')
                print(f'Epoch [{epoch + 1}/{num_epochs}], saved')
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= patience:
                print("Early stopping triggered! Training stopped.")
                break

    epochs = range(1, len(val_precision_macro_history) + 1)

    # 绘制并保存训练和验证的损失和其他指标的图形
    plt.figure(figsize=(20, 10))
    plt.plot(epochs, val_accuracy_history, label='Validation Accuracy')
    plt.plot(epochs, val_precision_macro_history, label='Validation macro Precision')
    plt.plot(epochs, val_recall_macro_history, label='Validation macro Recall')
    plt.plot(epochs, val_f1_macro_history, label='Validation macro F1 Score')

    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Evaluation Metrics')
    plt.legend()
    plt.savefig(
        f'/home/ubuntu/Data/hyw/filed-road-wheatData/filed-road-wheatData/wheatModelCode/wheatModelCode7--data_pro_all_point/train-valid-(train6)-6(14)/10/metrics_plot1_{fold_num}.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.plot(epochs, train_loss_avg_history, label='Average Training Loss')
    plt.plot(epochs, val_loss_avg_history, label='Average Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f'/home/ubuntu/Data/hyw/filed-road-wheatData/filed-road-wheatData/wheatModelCode/wheatModelCode7--data_pro_all_point/train-valid-(train6)-6(14)/10/Loss_Graph1_{fold_num}.png')
    plt.close()

    print("Model training completed.")


seed_id = 3407  # 设置随机种子以确保实验的可重复性
torch.manual_seed(seed_id)
torch.cuda.manual_seed_all(seed_id)
random.seed(seed_id)
np.random.seed(seed_id)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 定义模型，使用 DataParallel 来支持多 GPU 训练，将模型移至 GPU，调用 train 函数进行训练。
model = ConvTEModel(num_classes=2)
# model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
#model = nn.DataParallel(model, device_ids=[4, 5])
model = nn.DataParallel(model, device_ids=[6, 7])

model = model.to('cuda:6')
fold_num =9 # the number of fold is 0-9
# fold_num = 10 # 10 is code test

train(model, num_epochs=400, batch_size=None)  # 调用 train 函数进行训练
