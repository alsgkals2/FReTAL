

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
        img = np.array(img)
        img = torch.tensor(img)
        if self.transform:
            img = self.transform(img)
        print(img, _target)
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
        
        
def test_respectively(test_loader, model, criterion, epoch):
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