import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
import yaml
import sys
sys.path.append('/opt/ml/workspace/script')

import numpy as np
import torch
from timm.scheduler.step_lr import StepLRScheduler
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_f1_score

from util.loss import create_criterion
import wandb

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 


class CustomNamespace(argparse.Namespace):
    def __init__(self, config):
        for key, value in config.items():
            if isinstance(value, dict):
                setattr(self, key, CustomNamespace(value))
            else:
                setattr(self, key, value)
                
    def flatten_namespace(self, prefix=''):
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, CustomNamespace):
                config_dict.update(value.flatten_namespace(prefix=f"{prefix}{key}."))
            else:
                config_dict[f"{prefix}{key}"] = value
        return config_dict
          
      
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping")
        else:
            self.best_loss = val_loss
            self.counter = 0
        
    

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    seed_everything(args.train.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))


    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"DEVICE : {device}")


    # -- dataset
    dataset_module = getattr(import_module("util.dataset"), args.dataset.module)
    dataset = dataset_module(data_dir, args.dataset.age_split)
    num_classes = dataset.num_classes
    
    # -- augmentation
    transform_module = getattr(import_module("util.augmentation"), args.augmentation.module)
    transform = transform_module(
        resize=args.augmentation.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)
    
    
    # -- data_loader
    train_set, val_set = dataset.split_dataset(args.dataset.oversampling, args.dataset.oversample_category, args.dataset.oversample_weight)
    
    train_loader = DataLoader(
        train_set,
        batch_size=args.train.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.train.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True
    )
    
    # -- model
    model_module = getattr(import_module("model"), args.model.module)
    model = model_module(num_classes=num_classes, backbone=args.model.backbone).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.loss.type)
    age_weights = torch.FloatTensor(train_set.get_class_weight()).to(device)
    print(f"age label loss weight : {age_weights.tolist()}")
    age_criterion = create_criterion(args.loss.type, weight=age_weights)
    
    opt_module = getattr(import_module("torch.optim"), args.optimizer.type)
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.optimizer.lr,
        weight_decay=5e-4
    )
    
    scheduler = StepLRScheduler(
        optimizer,
        decay_t=args.scheduler.lr_decay_step,
        decay_rate=0.5,
        warmup_lr_init=2e-08,
        warmup_t=5,
        t_in_epochs=False,
    )  # warmup 적용을 위해 fastai team에서 만든 timm library의 scheduler


    # -- logging
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(args.flatten_namespace(), f, ensure_ascii=False, indent=4)
    
    
    # -- Train  
    best_val_acc = 0
    best_val_loss = np.inf
    best_val_targets = []
    best_val_preds = []
    
    early_stopping = EarlyStopping(patience=args.train.early_stop_round, delta=0.0)
    
    for epoch in range(args.train.epochs):
        # train loop
        model.train()
        
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, (mask_labels, gender_labels, age_labels) = train_batch
            labels = mask_labels * 6 + gender_labels * 3 + age_labels

            inputs = inputs.to(device)
            mask_labels = mask_labels.to(device)
            gender_labels = gender_labels.to(device)
            age_labels = age_labels.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            mask_outs, gender_outs, age_outs = torch.split(model(inputs), [3, 2, 3], dim=1)
            preds = torch.argmax(mask_outs, dim=-1) * 6 + torch.argmax(gender_outs, dim=-1) * 3 + torch.argmax(age_outs, dim=-1)
            
            mask_loss = criterion(mask_outs, mask_labels)
            gender_loss = criterion(gender_outs, gender_labels)
            age_loss = age_criterion(age_outs, age_labels)
            
            loss = mask_loss * args.loss.class_weight[0] + gender_loss * args.loss.class_weight[1] + age_loss * args.loss.class_weight[2] 
            
            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.train.log_interval == 0:
                train_loss = loss_value / args.train.log_interval
                train_acc = matches / args.train.batch_size / args.train.log_interval
                train_f1score = multiclass_f1_score(preds, labels, num_classes=18)
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.train.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || training f1score {train_f1score:4.2} || lr {current_lr}"
                )

                if args.wandb.logging:
                    wandb.log({"train_acc" : train_acc, "train_loss" : train_loss, "train_f1score" : train_f1score})
                
                loss_value = 0
                matches = 0

        try:
            scheduler.step()
        except:
            scheduler.step_update(epoch + 1)  # timm


        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            target = []
            pred = []
            
            for val_batch in val_loader:
                inputs, (mask_labels, gender_labels, age_labels) = val_batch
                labels = mask_labels * 6 + gender_labels * 3 + age_labels
                
                inputs = inputs.to(device)
                mask_labels = mask_labels.to(device)
                gender_labels = gender_labels.to(device)
                age_labels = age_labels.to(device)
                labels = labels.to(device)

                mask_outs, gender_outs, age_outs = torch.split(model(inputs), [3, 2, 3], dim=1)
                preds = torch.argmax(mask_outs, dim=-1) * 6 + torch.argmax(gender_outs, dim=-1) * 3 + torch.argmax(age_outs, dim=-1)
                
                target.extend(list(labels.to('cpu')))
                pred.extend(list(preds.to('cpu')))

                mask_loss = criterion(mask_outs, mask_labels)
                gender_loss = criterion(gender_outs, gender_labels)
                age_loss = criterion(age_outs, age_labels)
                loss = mask_loss + gender_loss + age_loss
                
                loss_item = loss.item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_loader) / args.train.valid_batch_size
            val_f1score = multiclass_f1_score(torch.tensor(pred), torch.tensor(target), num_classes=18)
            best_val_loss = min(best_val_loss, val_loss)
            early_stopping(val_loss)
            
            if early_stopping.early_stop:
                break
            
            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model, f"{save_dir}/best.pth")
                best_val_acc = val_acc
                val_f1score = val_f1score
                best_val_preds = pred
                best_val_targets = target
                
            torch.save(model, f"{save_dir}/last.pth")
            
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2}, f1score: {val_f1score:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}, best f1score: {val_f1score:4.2} \n"
            )
            if args.wandb.logging:
                wandb.log({"val_acc" : val_acc, "val_loss" : val_loss, "val_f1score" : val_f1score})
    
    if args.wandb.logging:
        wandb.sklearn.plot_confusion_matrix(best_val_targets, best_val_preds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    args = parser.parse_args()

    # Yaml 파일 로드
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(config)    
    args = CustomNamespace(config)
    
    if args.wandb.logging:
        wandb.login()
        wandb.init(
            entity=args.wandb.entity,
            project=args.wandb.project,
            name=args.name,
            config=args.flatten_namespace(),
            save_code=False
        )
    
    data_dir = args.dataset.path
    model_dir = args.model.save_path

    train(data_dir, model_dir, args)
