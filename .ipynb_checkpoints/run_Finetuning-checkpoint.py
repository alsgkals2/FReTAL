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

#LOSS-------------------
def loss_fn_kd(outputs, labels, teacher_outputs, KD_T=20, KD_alpha=0.5):
    KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/KD_T,dim=1),
                             F.softmax(teacher_outputs/KD_T,dim=1) * KD_alpha*KD_T*KD_T) +\
        F.cross_entropy(outputs, labels) * (1. - KD_alpha)
    return KD_loss

# L2-reg & L2-norm
def reg_cls(model):
    l2_cls = torch.tensor(0.).cuda()
    for name, param in model.named_parameters():
        if name.startswith('last_linear'):
            l2_cls += 0.5 * torch.norm(param) ** 2
    return l2_cls

def reg_l2sp(model):
    sp_loss = torch.tensor(0.).cuda()
    for name, param in model.named_parameters():
        if not name.startswith('last_linear'):
            sp_loss += 0.5 * torch.norm(param - teacher_model_weights[name]) ** 2
    return sp_loss
#-------------------
     
          
def test(val_loader, model, criterion, epoch):
    global best_acc
    correct, total =0,0
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    arc = AverageMeter()
    main_losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()

    end = time.time()
    with torch.no_grad():
            # measure data loading time
            data_time.update(time.time() - end)
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss_main = criterion(outputs, targets)
            loss_cls = 0
            loss_sp = 0
            loss = loss_main
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += len(targets)
            losses.update(loss.data.tolist(), inputs.size(0))
            main_losses.update(loss_main.tolist(), inputs.size(0))
            top1.update(correct/total, inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
    print(
        'Test | Loss:{loss:.4f} | MainLoss:{main:.4f} | top:{top:.4f}'.format(loss=losses.avg, main=main_losses.avg, top = top1.avg))
    return (losses.avg, arc.avg, top1.avg)
teacher_model, student_model = teacher_model.cuda(), student_model.cuda()
early_stopping = EarlyStopping(patience=10, verbose=True)
best_acc,epochs=0, 100
sys.stdout = open(save_path+'/{}_{}_sha_LQHQ_littleaug_{}_nobright.txt'.format(name_source,name_target,name_saved_file),'a')

print('save path : {} / epochs : {}'.format(save_path,epochs))
is_best_acc = False
amp=True
print('amp is {} / freeze mode is {}'.format(amp, use_freezing))
print(train_aug)
scaler = GradScaler()
for epoch in range(epochs):
    running_loss = []
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
        r = np.random.rand(1)
        if r > 0.8:
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            tt = targets[rand_index]
            boolean = targets != tt
            if True in boolean:
                rand_index = rand_index[boolean]
                lam = np.random.beta(0.5,0.5)
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[boolean, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
        #############################
        list_features_std = [[],[]]
        outputs = student_model(inputs)
        loss_main = criterion(outputs, targets)
        sne_loss=0
        optimizer.zero_grad()
        loss = loss_main
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += len(targets)
        running_loss.append(loss_main.cpu().detach().numpy())
        
    print("Epoch: {}/{} - CE_Loss: {:.4f} | ACC: {:.4f}".format(epoch+1, epochs, np.mean(running_loss), correct / total))
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
        print("-----EARLY STOPPED-----")
        exit()
    

