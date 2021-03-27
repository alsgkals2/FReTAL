import os
import torch
import shutil
import numpy as np
import os
def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch, opt):
    lr_set = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    lr_list = opt.schedule.copy()
    lr_list.append(epoch)
    lr_list.sort()
    idx = lr_list.index(epoch)
    opt.lr *= lr_set[idx]
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.lr
        
#mhkim add-temp
def save_arr_acc_loss(list_acc,list_real_acc,list_fake_acc,list_loss,
                      list_val_acc,list_val_real_acc,list_val_fake_acc,list_val_loss,
                      path):
    list_final,list_val_final=[],[]
    
    list_final.append(list_acc)
    list_final.append(list_real_acc)
    list_final.append(list_fake_acc)
    list_final.append(list_loss)
    list_val_final.append(list_val_acc)
    list_val_final.append(list_val_real_acc)
    list_val_final.append(list_val_fake_acc)
    list_val_final.append(list_val_loss)
    train_dir = path + '_train'
    val_dir = path + '_val'

    np.save(train_dir,list_final)
    np.save(val_dir,list_val_final)