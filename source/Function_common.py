import os
import os.path
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
class CustumDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform
    
    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path = self.data[idx]
        img = Image.open(path)
        img = img.convert('RGB')
        
#         if not torch.is_tensor(img):
#             img = np.array(img)
#             img = torch.tensor(img)
        if self.transform:
            img = self.transform(img)
        return img, self.target[idx]

def rand_bbox_custum(size, lam):
    W = size[2]
    H = size[3]
    bbx1,bbx2,bby1,bby2=0,W,0,H
    cut_rat = np.sqrt(1. - lam)
    ran_num = np.random.randint(100)
    # uniform
    if ran_num >50:
        cx = np.random.randint(W//2)
        cx = W//2 + cx
        bbx1 = np.clip(cx, 0, W)
    else:
        cy = np.random.randint(H//2)
        cy = W//2 + cy
        bby1 = np.clip(cy, 0, H)
    return bbx1, bby1, bbx2, bby2

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def save_checkpoint_for_unlearning(state, checkpoint, filename='checkpoint.pt',
                                   best_filename = 'student_model_best_acc.pt',
                                   cnt=0, isAcc=False):
    if isAcc:
        torch.save(state, os.path.join(checkpoint,best_filename))
        
def Test(val_loader, model, criterion, inputs,targets, epoch):
    global best_acc
    correct, total =0,0
    losses = AverageMeter()
    arc = AverageMeter()
    main_losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    model.cuda()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
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
    print(
        'Test | Loss:{loss:.4f} | MainLoss:{main:.4f} | top:{top:.4f}'.format(loss=losses.avg, main=main_losses.avg, top = top1.avg))
    return (losses.avg, arc.avg, top1.avg)

def Eval(test_loader, model, criterion, epoch):
    model.eval()
    model = model.cuda()
    losses = AverageMeter()
    arc = AverageMeter()
    acc_real = AverageMeter()
    acc_fake = AverageMeter()
    acc_ = AverageMeter()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == targets).squeeze()
            acc = [0, 0]
            class_total = [0, 0]
            for i in range(len(targets)):
                label = targets[i]
                acc[label] += 1 if c[i].item() == True else 0
                class_total[label] += 1
            #top1 = accuracy(outputs.data, targets.data)
            losses.update(loss.data.tolist(), inputs.size(0))
            if (class_total[0] != 0):
                acc_real.update(acc[0] / class_total[0])
            if (class_total[1] != 0):
                acc_fake.update(acc[1] / class_total[1])
            auroc = roc_auc_score(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy()[:, 1])
            arc.update(auroc, inputs.size(0))

        print("Real Accuracy : {:.2f}".format(acc_real.avg*100))
        print("Fake Accuracy : {:.2f}".format(acc_fake.avg*100))
        print("Accuracy : {:.2f}".format((acc_real.avg+acc_fake.avg)/2*100))
        print("arc.avg : {:.2f}".format(arc.avg*100))
        print("")
        
def Make_DataLoader(rootpath_dataset,name_source, name_target, train_aug=None,val_aug=None,mode_FReTAL = False):
    train_dir = os.path.join(rootpath_dataset+'/TransferLearning',name_target+'/train/')

    #For Validataion
    source_dataset = os.path.join(rootpath_dataset,name_source)
    target_dataset = os.path.join(rootpath_dataset,name_target)
    test_source_dir = os.path.join(source_dataset,'test')
    test_target_dir = os.path.join(target_dataset,'test')
    val_source_dir = os.path.join(source_dataset, 'val')
    val_target_dir = os.path.join(target_dataset, 'val')
    #check the paths
    if not(os.path.exists(test_source_dir) and os.path.exists(test_target_dir) and os.path.exists(val_source_dir) and os.path.exists(val_target_dir)) :
        print("check the paths")
        return
    print("DATASET PATHS")
    print(val_source_dir)
    print(test_source_dir)
    print(val_target_dir)
    print(test_target_dir)

                
    train_target_loader, train_target_loader_forcorrect = None,None
    train_target_dataset = datasets.ImageFolder(train_dir,transform=None)
    train_target_dataset = CustumDataset(np.array(train_target_dataset.samples)[:,0],np.array(train_target_dataset.targets),train_aug)
    train_target_loader = DataLoader(train_target_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    if mode_FReTAL : train_target_loader_forcorrect = DataLoader(train_target_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
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

    dic = {'train_target':train_target_loader,'val_source':val_source_loader,'test_source':test_source_loader,
          'val_target':val_target_loader, 'test_target':test_target_loader}
    dic_FReTAL = {'train_target_dataset':train_target_dataset ,'train_target_forCorrect':train_target_loader_forcorrect}
    return dic, dic_FReTAL
    #     return train_target_loader,val_source_loader,val_target_loader,test_source_loader, test_target_loader,train_target_loader_forcorrect
    
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