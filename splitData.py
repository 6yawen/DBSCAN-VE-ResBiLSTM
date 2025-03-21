import pickle
import os
import shutil



def spilt():
    
    path = '/home/ubuntu/Data/hyw/filed-road-wheatData/filed-road-wheatData/weights/weights/wheat_1_data_split.pkl'
    
    Kfold = 10  #定义了 Kfold 为10，表示要将数据划分为10个折叠，用于交叉验证。
    with open(path, 'rb') as file: 
        data = pickle.load(file, encoding='latin1')  
        train = data['train'] 
        valid = data['valid']
        test = data['test']

        print('train:', len(train))
        print('valid:', len(valid))
        print('test:', len(test))

        for i in range(Kfold):
            fold_path=spilt_path+"/"+str(i)
            if not os.path.exists(fold_path):
                os.mkdir(fold_path)
            if not os.path.exists(fold_path+"/train"):
                os.mkdir(fold_path+"/train")
            if not os.path.exists(fold_path+"/valid"):
                os.mkdir(fold_path+"/valid")
            if not os.path.exists(fold_path+"/test"):
                os.mkdir(fold_path+"/test")

            for j in train[i]:
                shutil.copyfile(files_path+"/"+j,fold_path+"/train/"+j)
            for j in valid[i]:
                shutil.copyfile(files_path + "/" + j, fold_path + "/valid/" + j)
            for j in test[i]:
                shutil.copyfile(files_path + "/" + j, fold_path + "/test/" + j)
            print('len train:',len(train[i]))
            print('train:',train[i])
            print('len valid:',len(valid[i]))
            print('valid:',valid[i])
            print('len test:',len(test[i]))
            print('test:',test[i])
            print('+++++++++++++++++++++++++++++++++++++++++++++++')
if __name__=="__main__":
    files_path = "/home/ubuntu/Data/hyw/filed-road-wheatData/filed-road-wheatData/calculateData/wheat/" #定义数据文件的路径
    spilt_path = files_path + "_10fold" 
    if not os.path.exists(spilt_path):
        os.mkdir(spilt_path)
    spilt() 
