

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pandas as pd
import time
from scipy import interp
import utils.tools as utils
from sklearn.model_selection import StratifiedKFold
import argparse
import dgl
from sklearn.preprocessing import StandardScaler
#from layers import GraphConvolution
from dgl.nn.pytorch import GraphConv
from sklearn.metrics import roc_curve, auc
 

#parser = argparse.ArgumentParser()
#parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
#parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
#parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
#parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
#parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
#parser.add_argument('--patience', type=int, default=100, help='Patience')

#args = parser.parse_args()
#args.cuda = not args.no_cuda and torch.cuda.is_available()

def get_shuffle(dataset,label):    
    index = [i for i in range(len(label))]
    np.random.shuffle(index)
    dataset = dataset[index]
    label = label[index]
    return dataset,label  

def to_categorical(y, nb_classes=None):
    y = np.array(y, dtype='int')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1
    return Y




data_=pd.read_csv(r'Elastic_RONGHE.csv')
data1=np.array(data_)
data=data1[:,1:]
[m1,n1]=np.shape(data)
trans = StandardScaler()
data = trans.fit_transform(data)
label1=np.ones((int(),1))
label2=np.zeros((int(),1))
labels=np.append(label1,label2)
shu=data
y=labels
X,y=get_shuffle(shu,labels)
features = torch.FloatTensor(X)
y = torch.tensor(y).long()
labels = torch.squeeze(y)


skf= StratifiedKFold(n_splits=5,shuffle=True,random_state = 42)
dur = []

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)    #max(1)返回每一行中最大值的那个元素所构成的一维张量，且返回对应的一维索引张量（返回最大元素在这一行的列索引）
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
    
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hid_feats)
        self.conv2 = GraphConv(hid_feats, out_feats)
        self.dropout =  nn.Dropout(p=0.5)
    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = F.relu(h)
        h = self.conv2(g, h)
        output = F.softmax(h, dim=1)
        return output

    



best_accuracy = 0  
for train_index,test_index in skf.split(X,y):
    
     net = GCN(in_feats=332,hid_feats=32,out_feats=2)
     optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
     #把训练集图表示
     features11 = torch.FloatTensor(X[train_index])
     g11 = dgl.knn_graph(features11, 5, algorithm='bruteforce', dist='cosine')

     #把验证集图表示
     features22 = torch.FloatTensor(X[test_index])
     g22 = dgl.knn_graph(features22, 5, algorithm='bruteforce', dist='cosine')

     for epoch in range(30):
         #训练模型，训练集用于训练    
         net.train()
         optimizer.zero_grad()
         t0 = time.time()
         logits = net(g11,features11)
         loss = nn.CrossEntropyLoss()
         logits = logits.float()
         loss_train =loss(logits, labels[train_index])
         acc_train = accuracy(logits, labels[train_index])
         loss_train.backward()
         optimizer.step()        
        # 验证模型，验证集用于验证，不参与参数更新
         net.eval()
         with torch.no_grad():
            logits22=net(g22,features22) 
            loss_val = loss(logits22, labels[test_index])
            acc_val = accuracy(logits22, labels[test_index])             
         dur.append(time.time() - t0)
         if acc_val > best_accuracy:
             best_accuracy = acc_val
     net = torch.load('')
     with torch.no_grad():
         a=net(g22,features22)   
         probas = F.softmax(a, dim=1)   # 按行SoftMax,行和为1
         y_class = np.argmax(probas.detach().numpy(), axis=1)
         acc, precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(y_class), y_class,labels[test_index])
         fpr, tpr, thresholds = roc_curve(labels[test_index], probas.detach().numpy()[:, 1])
         roc_auc = auc(fpr, tpr)
         print('NB:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
          % (acc, precision,npv, sensitivity, specificity, mcc,f1, roc_auc))
#测试
net = torch.load()
log_str = 'test results: acc=%f, precision=%f, npv=%f, sensitivity=%f, specificity=%f, mcc=%f, f1=%f, roc_auc=%f' % (acc, precision, npv, sensitivity, specificity, mcc, f1, roc_auc)

