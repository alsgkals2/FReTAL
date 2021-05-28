import sys
import os
import os.path
import time
import torch
import random
import numpy as np
import pandas as pdb
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.backends.cudnn as cudnn
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import torchvision
import torchvision.datasets as datasets
import copy
from torch.nn import functional as F
import xception_origin
from torch.utils.data import DataLoader
from common import 
from PIL import Image
import cv2

#MUST WRITE THE ARGUMENTS [1]~[5]
try:
    num_gpu = sys.argv[1]
    name_source = sys.argv[2]
    name_target = sys.argv[3]
    name_saved_file = sys.argv[4]
    use_freezing = sys.argv[5]
except:
    print("Please check the arguments")
    print("[number_gpu] [source data name] [target data name] [save file name] ['True' if you want to 'freeze the some layers of student model'] [Write the 'folder name' if you want to devide for more detail]
try:
    name_saved_folder = sys.argv[6]
except:
    name_saved_folder= ''
lr = 0.05 # YOU CAN CHANGE THE LEARNING RATE
KD_alpha = 0.5 # YOU CAN CHANGE
num_class = 2
num_store_per = 5
          
print('KD_alpha is ',KD_alpha)
print('gpu num is' , num_gpu)
os.environ['CUDA_VISIBLE_DEVICES'] =str(num_gpu)

random_seed = 2020
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

save_path = './What path you want to save/'
try:
    if not os.path.isfile(save_path):
        os.makedirs(save_path)
except OSError:
    pass

#train & valid
source_dataset = '/rootpath_dataset/'+name_source
target_dataset = '/rootpath_dataset/'+name_target
train_dir = '/rootpath_dataset/path_TransferLearning/'+name_target
test_source_dir = os.path.join(source_dataset,'test')
test_target_dir = os.path.join(target_dataset,'test')
val_source_dir = os.path.join(source_dataset, 'val')
val_target_dir = os.path.join(target_dataset, 'val')


train_aug = transforms.Compose([
    transforms.Resize(128),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])

val_aug = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])

import xception_origin
train_target_dataset = datasets.ImageFolder(train_dir,transform=None)
train_target_dataset = CustumDataset(np.array(train_target_dataset.samples)[:,0],np.array(train_target_dataset.targets),train_aug)
train_target_loader = DataLoader(train_target_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
train_target_loader_forcorrect = DataLoader(train_target_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

val_target_loader = DataLoader(datasets.ImageFolder(val_target_dir, val_aug),
                               batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
val_source_loader = DataLoader(datasets.ImageFolder(val_source_dir, val_aug),
                               batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

#test - target
test_source_dataset = datasets.ImageFolder(test_source_dir,transform=None)
cond = np.array(test_source_dataset.targets)==1
test_source_dataset.targets = np.array(test_source_dataset.targets)
test_source_dataset.samples = np.array(test_source_dataset.samples)
print(test_source_dataset.targets.shape)
test_source_dataset = CustumDataset(test_source_dataset.samples[:,0],test_source_dataset.targets,train_aug)
test_source_loader = DataLoader(test_source_dataset,
                         batch_size=50, shuffle=True, num_workers=4, pin_memory=True)

#test - source
test_target_dataset = datasets.ImageFolder(test_target_dir,transform=None)
cond = np.array(test_target_dataset.targets)==1
test_target_dataset.targets = np.array(test_target_dataset.targets)
test_target_dataset.samples = np.array(test_target_dataset.samples)
print(test_target_dataset.targets.shape)
test_target_dataset = CustumDataset(test_target_dataset.samples[:,0],test_target_dataset.targets,train_aug)
test_target_loader = DataLoader(test_target_dataset,
                         batch_size=50, shuffle=True, num_workers=4, pin_memory=True)

teacher_model, student_model = None,None
try:
    path_pretrained = 'path_path/'+str(sys.arg[2])
    teacher_model = xception_origin.xception(num_classes=2, pretrained='')
    checkpoint =torch.load(path_pretrained+'/model_best_accuracy.pth.tar')
    teacher_model.load_state_dict(checkpoint['state_dict'])

    student_model = xception_origin.xception(num_classes=2,pretrained='')
    checkpoint =torch.load(path_pretrained+'/model_best_accuracy.pth.tar')
    student_model.load_state_dict(checkpoint['state_dict'])
except:
    print("Please check the path")
    
teacher_model.eval()
student_model.train()
teacher_model.cuda()
student_model.cuda()

#FREASING THE TEACHER MODEL
teacher_model_weights = {}
for name, param in teacher_model.named_parameters():
    teacher_model_weights[name] = param.detach()        

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(student_model.parameters(), lr=lr, momentum=0.1)

list_correct = func_correct(teacher_model.cuda(),train_target_loader_forcorrect)
correct_loaders,list_ratio_loader = GetSplitLoaders_BinaryClasses(list_correct,train_target_dataset)
list_ratio_loader = torch.tensor(list_ratio_loader)
          
# FIXED THE AVG OF FEATURES. IT IS FROM A TEACHER MODEL
list_features = GetListTeacherFeatureFakeReal(teacher_model,correct_loaders)
list_features = np.array(list_features)
list_ratio_loader = np.array(list_ratio_loader)

teacher_model, student_model = teacher_model.cuda(), student_model.cuda()

early_stopping = EarlyStopping(patience=10, verbose=True)
best_acc,epochs=0, 100
print('epochs={}'.format(epochs))
is_best_acc = False

for epoch in range(epochs):
    running_loss = []
    running_loss_kd = []
    running_loss_other = []
    correct,total = 0,0
    teacher_model.eval()
    student_model.train()

    losses = AverageMeter()
    arc = AverageMeter()
    cls_losses = AverageMeter()
    sp_losses = AverageMeter()
    main_losses = AverageMeter()
    alpha = AverageMeter()
    real_acc = AverageMeter()
    fake_acc = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(train_target_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        sne_loss = None
        ##tgd aug
        r = np.random.rand(1)
        
        if r > 0.8:
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            tt = targets[rand_index]
            boolean = targets != tt #THIS IS ALWAYS ATTACHING THE OPPOSITED THE 'SMALL PIECE OF A DATA'
            if True in boolean:
                rand_index = rand_index[boolean]
                lam = np.random.beta(0.5,0.5)
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[boolean, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]

        correct_loader_std,_ = correct_binary(student_model.cuda(), inputs, targets)
        list_features_std = [[] for i in range(num_class)]
        optimizer.zero_grad()
        for j in range(num_store_per):
            for i in range(num_class):
                feat = GetFeatureMaxpool(student_model,correct_loader_std[j][i])
                if(list_features[i][j]==0):continue
                feat = feat-torch.tensor(list_features[i][j]).cuda()
                feat = torch.pow(feat.cuda(),2)
                list_features_std[i].append(feat)
          
        teacher_outputs = teacher_model(inputs)
        outputs = student_model(inputs)
        loss_main = criterion(outputs, targets)
        loss_kd = loss_fn_kd(outputs,targets,teacher_outputs)
        sne_loss=0
        for fs in list_features_std:
            for ss in fs:
                if ss.requires_grad:
                    sne_loss += ss
        loss = loss_main + loss_kd + sne_loss
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += len(targets)
        running_loss.append(loss_main.cpu().detach().numpy())
        running_loss_kd.append(loss_kd.cpu().detach().numpy())
        try:
            running_loss_other.append(sne_loss.cpu().detach().numpy())
        except AttributeError:
            running_loss_other.append(sne_loss)  
    print("Epoch: {}/{} - CE_Loss: {:.4f} | KD_Loss: {:.4f} | OTHER_LOSS: {:.4f} | ACC: {:.4f}".format(epoch+1, epochs, np.mean(running_loss), np.mean(running_loss_other),  np.mean(running_loss_kd), correct / total))
    
    #validataion
    test_loss, test_auroc, test_acc = test(val_target_loader, student_model, criterion, epoch)
    source_loss, source_auroc, source_acc = test(val_source_loader, student_model, criterion, epoch)

    is_best_acc = test_acc + source_acc > best_acc
    best_acc = max(test_acc + source_acc, best_acc)
    is_epoch20per = False
    if (epoch+1)%10 ==0 or is_best_acc:
        is_best_acc = True
        best_acc = max(correct / total,best_acc)
        save_checkpoint_for_unlearning({
            'epoch': epoch + 1,
            'state_dict': student_model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict()
        }, cnt=epoch, isAcc=is_best_acc,
            checkpoint=save_path,
            best_filename = '{}_epoch_{}.pth.tar'.format(name_saved_file,epoch+1 if (epoch+1)%10==0 else ''))
    is_best_acc = False
    early_stopping(best_acc)
    if early_stopping.early_stop:
        print("early stoped-----------------------------")
        exit()
