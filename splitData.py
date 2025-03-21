import pickle
import os
import shutil


#定义 spilt() 函数：该函数用于将数据集划分为训练集、验证集和测试集，并进一步分成多个折叠（fold）
def spilt():
    #path 变量指定了存储数据的文件路径，这里是一个 .pkl 文件
    path = '/home/ubuntu/Data/hyw/filed-road-wheatData/filed-road-wheatData/weights/weights/wheat_1_data_split.pkl'
    #path = '/home/ubuntu/Data/zxy/gnss/weights/paddy_data_split.pkl'
    #path = '/home/ubuntu/Data/Bcx/paper1/weights/corn_data_split.pkl'
    Kfold = 10  #定义了 Kfold 为10，表示要将数据划分为10个折叠，用于交叉验证。
    with open(path, 'rb') as file:  #打开 .pkl 文件：以只读二进制模式 ('rb') 打开 path 指定的文件。
        data = pickle.load(file, encoding='latin1')  #使用 pickle.load() 函数读取 .pkl 文件中的数据。encoding='latin1' 用于解码数据。
        train = data['train'] #从读取的 data 字典中提取训练集 (train)、验证集 (valid) 和测试集 (test)。
        valid = data['valid']
        test = data['test']

        print('train:', len(train))
        print('valid:', len(valid))
        print('test:', len(test))

        for i in range(Kfold):#遍历每个折叠（共10个），为每个折叠创建一个文件夹，路径为 spilt_path（拼接折叠的编号 i）
            fold_path=spilt_path+"/"+str(i)#如果指定路径 fold_path 不存在，则创建相应的文件夹。接着为 train、valid 和 test 数据集创建子文件夹。
            if not os.path.exists(fold_path):
                os.mkdir(fold_path)
            if not os.path.exists(fold_path+"/train"):
                os.mkdir(fold_path+"/train")
            if not os.path.exists(fold_path+"/valid"):
                os.mkdir(fold_path+"/valid")
            if not os.path.exists(fold_path+"/test"):
                os.mkdir(fold_path+"/test")

            for j in train[i]:#将当前折叠中的训练、验证和测试集的文件从 files_path 复制到新创建的折叠文件夹中。shutil.copyfile() 用于将文件从源路径复制到目标路径
                shutil.copyfile(files_path+"/"+j,fold_path+"/train/"+j)
            for j in valid[i]:
                shutil.copyfile(files_path + "/" + j, fold_path + "/valid/" + j)
            for j in test[i]:
                shutil.copyfile(files_path + "/" + j, fold_path + "/test/" + j)
            print('len train:',len(train[i]))#输出每个折叠中的训练集、验证集、测试集的样本数量及其内容
            print('train:',train[i])
            print('len valid:',len(valid[i]))
            print('valid:',valid[i])
            print('len test:',len(test[i]))
            print('test:',test[i])
            print('+++++++++++++++++++++++++++++++++++++++++++++++')
if __name__=="__main__":
    files_path = "/home/ubuntu/Data/hyw/filed-road-wheatData/filed-road-wheatData/calculateData6(7--V2.0)/wheat/" #定义数据文件的路径
    spilt_path = files_path + "_10fold" #为存储折叠数据集创建一个新的路径（在原路径上拼接 _10fold）。
    if not os.path.exists(spilt_path):
        os.mkdir(spilt_path)
    spilt() #开始数据集的划分