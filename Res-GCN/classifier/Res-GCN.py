
import tensorflow as tf
import pickle
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
from dgl.nn.pytorch import GraphConv
from sklearn.metrics import roc_curve, auc

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



data_=pd.read_csv(r'')
data=np.array(data_)
data=data[:,1:]
[m1,n1]=np.shape(data)
trans = StandardScaler()
data = trans.fit_transform(data)
label1=np.ones((int(m1/2),1))
label2=np.zeros((int(m1/2),1))
labels=np.append(label1,label2)
shu=data
y=labels
X,y=get_shuffle(shu,labels)
features = torch.FloatTensor(X)
y = torch.tensor(y).long()
labels = torch.squeeze(y)
skf= StratifiedKFold(n_splits=2,shuffle=False)
dur = []

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)    #max(1)返回每一行中最大值的那个元素所构成的一维张量，且返回对应的一维索引张量（返回最大元素在这一行的列索引）
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class GCNNet(nn.Module):
    def __init__(self, num_features, hidden_dim, out_feats):
        super(GCNNet, self).__init__()
        self.conv1 = GraphConv(num_features, hidden_dim)
        nn.Dropout(0.5)
        self.conv2 = GraphConv(hidden_dim, out_feats)
    def forward(self, edge_index,x):
        x = self.conv1(edge_index,x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        return x
          
class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        #残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(outchannel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(outchannel,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            #shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outchannel,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
            )
            
    def forward(self, x):
        out = self.left(x)
        #将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out
 
#实现ResNet-18模型
class ResNet(nn.Module):
    def __init__(self, ResBlock, num_classes=32):
        super(ResNet, self).__init__()
        self.inchannel =256
        self.conv1 = nn.Sequential(
            nn.Conv1d(294, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        )
        self.layer1 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer2 = self.make_layer(ResBlock, 64, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 32, 2, stride=2)        
        self.layer4 = self.make_layer(ResBlock, 64, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((64, 1))        
        self.fc = nn.Linear(32, num_classes)
        
    #这个函数主要是用来，重复同一个残差块    
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        #在这里，整个ResNet18的结构就很清晰了
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
      
class CombinedNet(nn.Module):
    def __init__(self):
        super(CombinedNet, self).__init__()
        self.gcn = GCNNet(num_features, hidden_dim, out_feats)
        self.dilated_conv = ResNet(ResBlock, num_classes)
    def forward(self,inputs,g):
        h = self.gcn(inputs,g)
        h = h.unsqueeze(-1)
        out = self.dilated_conv(h)
        return out


 
best_accuracy = 0  
sepscores = []
sepscores_ = []
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5
probas_cnn=[]
tprs_cnn = []
sepscore_cnn = []

for train_index,test_index in skf.split(X,y):
    
     net = CombinedNet()
     optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
     features11 = torch.FloatTensor(X[train_index])
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


