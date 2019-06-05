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

ckpt = '../../backup_main/latest.ckpt'
#ckpt = '../../backup_balanced/latest.ckpt'
#ckpt = '../../backup_balanced/best_top1_val_acc.ckpt'

checkpoint = torch.load(ckpt)
state_dict = checkpoint['state_dict']

cudnn.benchmark = True
net.load_state_dict(state_dict)
net.to('cuda')
net = nn.DataParallel(net)

validate_dataset = INAT(data_root,test_file,image_size, is_train=False)
validate_loader = DataLoader(validate_dataset,batch_size=batch,shuffle=False,num_workers=4,pin_memory=True)

net.eval()
f = open("submission.csv","w")
f.write('id,predicted\n')

with torch.no_grad():
    for i,(X,im_id,y,_) in enumerate(validate_loader):
        n = y.size(0)
        X = X.to('cuda')
        y = y.to('cuda')
        y_pred,feature_matrix,attention_map = net(X)

        print('Batch:',i)
        _, top_5_pred = y_pred.topk(5,1,True,True)
        top_5 = top_5_pred.data.cpu().numpy()
        img_id = im_id.data.cpu().numpy()
        
        for j in range(n):
            f.write(str(img_id[j]))
            f.write(',')
            
            for k in range(5):
                f.write(str(top_5[j][k]))
                f.write(' ')
            f.write('\n')

f.close()

