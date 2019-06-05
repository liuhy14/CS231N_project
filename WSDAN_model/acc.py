from dataset import *
from models import *
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import accuracy
import torch.nn as nn
import os
import numpy as np
import csv
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

data_root = os.path.join('..','..','dataset')
train_file = os.path.join('..','..','dataset','train2019.json')
val_file = os.path.join('..','..','dataset','val2019.json')
test_file = os.path.join('..','..','dataset','test2019.json')

image_size = (512,512)

num_attentions = 32
feature_net = inception_v3(pretrained=True)
num_classes = 1010
batch = 64

net = WSDAN(num_classes=num_classes, M=num_attentions,net=feature_net)

#ckpt = '../../backup_main/latest.ckpt'
ckpt = '../../backup_balanced/latest.ckpt'
#ckpt = '../../backup_balanced/best_top1_val_acc.ckpt'

checkpoint = torch.load(ckpt)
state_dict = checkpoint['state_dict']

cudnn.benchmark = True
net.load_state_dict(state_dict)
net.to('cuda')
net = nn.DataParallel(net)


validate_dataset = INAT(data_root,val_file,image_size, is_train=False)
validate_loader = DataLoader(validate_dataset,batch_size=batch,shuffle=False,num_workers=4,pin_memory=True)

top1 = np.zeros(num_classes)
top3 = np.zeros(num_classes)
top5 = np.zeros(num_classes)

total = np.zeros(num_classes)
net.eval()

with torch.no_grad():
    for i,(X,_,y,_) in enumerate(validate_loader):
        n = y.size(0)
        X = X.to('cuda')
        y = y.to('cuda')
        y_pred,feature_matrix,attention_map = net(X)

        values,indices = y_pred.max(1)
        _, top_3_pred = y_pred.topk(3,1,True,True)
        _, top_5_pred = y_pred.topk(5,1,True,True)

        print('Batch:',i)

        top_1 = indices.data.cpu().numpy()
        top_3 = top_3_pred.data.cpu().numpy()
        top_5 = top_5_pred.data.cpu().numpy()

        y_targ = y.data.cpu().numpy()
        
        #print('Accuracy: ',np.sum(top_1==y_targ)/n)
        for j in range(n):
            total[y_targ[j]] += 1;
            if (y_targ[j] == top_1[j]):
                top1[y_targ[j]] += 1;
            if (y_targ[j] in top_3[j]):
                top3[y_targ[j]] += 1;
            if (y_targ[j] in top_5[j]):
                top5[y_targ[j]] += 1;

pickle.dump(top1,open("top1.pkl","wb"))
pickle.dump(top3,open("top3.pkl","wb"))
pickle.dump(top5,open("top5.pkl","wb"))

#pickle.dump(total,open("total.pkl","wb"))

print("Top 1 Accuracy",sum(top1)/sum(total))
print("Top 3 Accuracy",sum(top3)/sum(total))
print("Top 5 Accuracy",sum(top5)/sum(total))



