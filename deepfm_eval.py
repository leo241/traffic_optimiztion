import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data

import time, json, datetime
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from deepfm_train import DeepFM
from sklearn.metrics import confusion_matrix,accuracy_score
from keras.utils import np_utils # 用于独热编码

if __name__ == '__main__':
    # hyper para
    weights_path = 'data/deepfm_best_0601.pth'

    data = pd.read_csv('data_0601.csv')
    dense_features = [ 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
       'Visibility(mi)',  'Wind_Speed(mph)',
       'Precipitation(in)',  'Amenity', 'Bump', 'Crossing',
       'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',
       'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop',
       ]
    sparse_features = ['Wind_Direction','Weather_Condition','Sunrise_Sunset'
            ,'Civil_Twilight','Nautical_Twilight','Astronomical_Twilight','month']
    target = ['Severity']

    ## 类别特征labelencoder
    for feat in tqdm(sparse_features):
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    ## 数值特征标准化
    for feat in tqdm(dense_features):
        mean = data[feat].mean()
        std = data[feat].std()
        data[feat] = (data[feat] - mean) / (std + 1e-12)  # 防止除零

    print(data.shape)

    train, valid = train_test_split(data, test_size=0.05, random_state=0)
    print(train.shape, valid.shape)

    valid_dataset = Data.TensorDataset(torch.LongTensor(valid[sparse_features].values),
                                       torch.FloatTensor(valid[dense_features].values),
                                       torch.FloatTensor(np_utils.to_categorical(valid['Severity'].values - 1, 4)))
    valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=4096, shuffle=False)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    cate_fea_nuniqs = [data[f].nunique() for f in sparse_features]
    model = DeepFM(cate_fea_nuniqs, nume_fea_size=len(dense_features),emb_size=64,
                 hid_dims=[512,128], num_classes=4, dropout=[0.0, 0.0])
    weights_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(weights_dict)
    model.to(device)
    # model.train()
    model.eval() #

    label_all = []
    pred_all = []
    with torch.no_grad():
        valid_labels, valid_preds = [], []
        for idx, x in tqdm(enumerate(valid_loader)):
            cate_fea, nume_fea, label = x[0], x[1], x[2]
            cate_fea, nume_fea,label = cate_fea.to(device), nume_fea.to(device),label.to(device)
            pred = model(cate_fea, nume_fea)
            pred2 =torch.argmax(pred,1).cpu().numpy().tolist()
            label2 = torch.argmax(label,1).cpu().numpy().tolist()
            # pred = model(cate_fea, nume_fea).reshape(-1).data.cpu().numpy().tolist()
            valid_preds.extend(pred2)
            valid_labels.extend(label2)
            label_all.append(label.cpu().numpy())
            pred_all.append(pred.cpu().numpy())
    ytrue = np.around(valid_labels)
    ypre = np.around(valid_preds)
    print('准确率：',accuracy_score(ytrue,ypre))
    print(confusion_matrix(ytrue,ypre))
    ytrue2 = np.concatenate(label_all)
    ypre2 = np.concatenate(pred_all)
    print('auc:',roc_auc_score(ytrue2, ypre2))