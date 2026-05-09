# kd_research/engine/trainer.py
import torch
import torch.nn as nn
import time
from typing import Optional
from utils.mixup import mixup_criterion

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

def train_one_epoch(
    model_s: torch.nn.Module,
    model_t: Optional[torch.nn.Module], 
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion_ce: torch.nn.Module,
    kd_module: Optional[torch.nn.Module] = None,
    mixup_fn = None,
    amp_enabled: bool = False,
    log_interval: int = 50
):
    model_s.train()
    if model_t:
        model_t.eval()

    try:
        scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    
    losses_m = AverageMeter()
    acc_m = AverageMeter()
    kd_loss_m = AverageMeter()
    
    def _get_fc_weight(model):
        if hasattr(model, "module"): model = model.module
        
        # 1. ResNet / ShuffleNet
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear): 
            return model.fc.weight
            
        # 2. ViT / Swin
        if hasattr(model, "head") and isinstance(model.head, nn.Linear): 
            return model.head.weight
            
        # 3. MobileNet / VGG / EfficientNet
        if hasattr(model, "classifier"):

            if isinstance(model.classifier, nn.Linear):
                return model.classifier.weight

            if isinstance(model.classifier, nn.Sequential):
                for layer in reversed(model.classifier):
                    if isinstance(layer, nn.Linear):
                        return layer.weight
        return None
    # ============================================================

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        if mixup_fn is not None:
            inputs, targets_a, targets_b, lam = mixup_fn(inputs, targets)
        
        optimizer.zero_grad()

        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            autocast_ctx = torch.amp.autocast('cuda', enabled=amp_enabled)
        else:
            autocast_ctx = torch.cuda.amp.autocast(enabled=amp_enabled)

        with autocast_ctx:
            output_s = model_s(inputs)
            
            output_t = None
            if model_t is not None:
                with torch.no_grad():
                    output_t = model_t(inputs)
            
            if mixup_fn is not None:
                loss_ce = mixup_criterion(criterion_ce, output_s, targets_a, targets_b, lam)
            else:
                loss_ce = criterion_ce(output_s, targets)
            
            loss_kd_part = 0.0
            if kd_module is not None and model_t is not None:
                w_s = _get_fc_weight(model_s)
                w_t = _get_fc_weight(model_t)
                

                if w_s is not None and w_t is not None:
                    kd_out = kd_module(output_s, output_t, w_s, w_t)
                    loss_kd_part = kd_out["loss_total"]
                    kd_loss_m.update(loss_kd_part.item(), inputs.size(0))
                else:

                    pass

            loss = loss_ce + loss_kd_part

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses_m.update(loss.item(), inputs.size(0))
        if mixup_fn is not None:
            acc1 = (output_s.argmax(1) == (targets_a if lam > 0.5 else targets_b)).float().mean() * 100
        else:
            acc1 = (output_s.argmax(1) == targets).float().mean() * 100
        acc_m.update(acc1.item(), inputs.size(0))

        if batch_idx % log_interval == 0:
            print(f"Train: [{batch_idx}/{len(data_loader)}] "
                  f"Loss: {losses_m.avg:.4f} "
                  f"Acc: {acc_m.avg:.2f} "
                  f"KD_Loss: {kd_loss_m.avg:.4f}")

    return losses_m.avg, acc_m.avg

@torch.no_grad()
def validate(model, data_loader, device, criterion):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

    return top1.avg, top5.avg, losses.avg

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res