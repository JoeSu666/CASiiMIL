import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


import argparse
import os
from os.path import join
import math
import random
import time
import mymodel.mymodel as model
import numpy as np
import glob
import shutil
import pandas as pd
from PIL import Image
import mydataset.mydataset as mydataset
import utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


parser = argparse.ArgumentParser(description='cam16 casii')
parser.add_argument('--model', default='', type=str,
                    help='path to the model')
parser.add_argument('--arch', default='casii', type=str,
                    help='architecture')
parser.add_argument('--data', default='cam16_ca', type=str,
                    help='dataset')     
parser.add_argument('--seed', default=7, type=int,
                    help='torch random seed')                                   
parser.add_argument('--split', default=42, type=int,
                    help='split random seed')
parser.add_argument('--splitrate', default=0.1, type=float,
                    help='train val split rate')   
parser.add_argument('--nfold', default=5, type=int,
                    help='num of folds')
parser.add_argument('--startfold', default=42, type=int,
                    help='startfold')                    
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    help='batch size')            
parser.add_argument('--inputd', default=1024, type=int,
                    help='input dim')      
parser.add_argument('--hd', default=512, type=int,
                    help='hidden layer dim')                                                       
parser.add_argument('--code', default='test', type=str,
                    help='exp code')                        
parser.add_argument('-t', '--threshold', default=0.5, type=float,
                    help='accuracy threshold') 
parser.add_argument('--pretrained', default='model_best.pth.tar', type=str, 
                    help='pretrained model for validate')    
                                                                                        
parser.add_argument('-k', default=5, type=float, help='top-k query predictions')
parser.add_argument('--tau', default=1, type=float, help='softmax temperature')
parser.add_argument('--keys', default='', type=str, help='keyset file')
parser.add_argument('--sc', default=None, type=float, help='use sparse coding')
parser.add_argument('--wu', default=5, type=int, help='warm up epoch')
parser.add_argument('--ql1', default=0.5, type=float, help='use botk query loss')
parser.add_argument('--ql2', default=0.5, type=float, help='use topk query loss')

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run(args):

    net = getattr(model, args.arch)(inputd=args.inputd, hd=args.hd, k=args.k, tau=args.tau)
    criterions = [nn.BCEWithLogitsLoss().cuda('cuda'), nn.BCEWithLogitsLoss().cuda('cuda')]

    val_dataset = getattr(mydataset, args.data)(train='val', keys=args.keys, split=args.split, splitrate=args.splitrate)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=32, pin_memory=True)

    test_dataset = getattr(mydataset, args.data)(train='test', keys=args.keys, split=args.split, splitrate=args.splitrate)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=32, pin_memory=True)


    print('load model from: ', join(args.save, args.pretrained))
    checkpoint = torch.load(join(args.save, args.pretrained), map_location="cpu")

    state_dict = checkpoint['state_dict']
    msg = net.load_state_dict(state_dict, strict=True)
    print(msg.missing_keys)
    net.cuda()
    
    val_metrics = validate(val_loader, net, criterions, args.threshold, test=False)
    test_metrics = validate(test_loader, net, criterions, args.threshold, test=True)

    return test_metrics, val_metrics


def validate(val_loader, model, criterions, threshold, test):
    criterion, criterion_q = criterions
    losses = utils.AverageMeter('Loss', ':.4e')

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):    
            images, keys, target = data
            images = images.cuda()
            keys = keys.cuda()
            target = target.cuda().float()
            output, _, topq, botq = model((images, keys))
            output = output.view(-1).float()

            if i == 0:
                outputs = output
                targets = target
            else:
                outputs = torch.cat((outputs, output), 0)
                targets = torch.cat((targets, target), 0)

            loss = criterion(output, target)
            if args.ql2:
                if target == 1:
                    qtarget = torch.ones(botq.shape).cuda().float()
                    loss += args.ql2 * criterion_q(topq, qtarget)
                # if target == 0:
            if args.ql1:
                qtarget = torch.zeros(botq.shape).cuda().float()
                loss += args.ql1 * criterion_q(botq, qtarget)
        losses.update(loss.item(), images.size(0))

    auc, acc, precision, recall, f1 = utils.eval_accuracy(outputs, targets, threshold)

    if not test:
        print(' ***Validation AUC {:.3f} ACC {:.3f} precision {:.3f} recall {:.3f} f1 {:.3f}'
            .format(auc, acc, precision, recall, f1))
    else:
        print(' ***Testing AUC {:.3f} ACC {:.3f} precision {:.3f} recall {:.3f} f1 {:.3f}'
            .format(auc, acc, precision, recall, f1))

    return auc, acc, precision, recall, f1, losses.avg, torch.sigmoid(outputs).cpu().numpy(), targets.cpu().numpy()


if __name__ == '__main__':
    args = parser.parse_args()
    seed_torch(args.seed)

    save_dir = './runs/'+'{}_wu{}_s{}_k{}_ql1-{}_ql2-{}'.format(args.code, args.wu, args.seed, args.k, args.ql1, args.ql2)

    val_aucs = []
    val_f1s = []
    aucs, accs, precisions, recalls, f1s = [], [], [], [], []
    results = {}
    testnamelist = glob.glob(join('../attention2minority/data/feats/cam16res', 'test', '*', '*.npy'))
    testnamelist = [name.split('/')[-1].split('.')[0] for name in testnamelist]

    for i in range(42, 42+args.nfold):       
        args.split = i       
        args.save = os.path.join(save_dir, str(args.split))                  
        test_metrics, val_metrics = run(args)
        aucs.append(test_metrics[0])
        accs.append(test_metrics[1])
        precisions.append(test_metrics[2])
        recalls.append(test_metrics[3])
        f1s.append(test_metrics[4])
        val_aucs.append(val_metrics[0])
        val_f1s.append(val_metrics[4])
        if i == 42:
            results['gts'] = test_metrics[-1]
        results[str(i)+'outputs'] = test_metrics[-2]
        
    results = pd.DataFrame(results, index=testnamelist)
    results.to_csv(join(save_dir, 'results.csv'))
    print('')
    print('========================================================')
    print('*Val AUC {:.3f} +- {:.3f}'.format(np.mean(val_aucs), np.std(val_aucs)))
    print('*Val f1 {:.3f} +- {:.3f}'.format(np.mean(val_f1s), np.std(val_f1s)))

    print('**Testing AUC {:.3f} +- {:.3f}, ACC {:.3f} +- {:.3f}, PRECISION {:.3f} +- {:.3f}, RECALL {:.3f} +- {:.3f}, F1 {:.3F} +- {:.3F}'
        .format(np.mean(aucs), np.std(aucs), np.mean(accs), np.std(accs), np.mean(precisions), np.std(precisions), 
        np.mean(recalls), np.std(recalls), np.mean(f1s), np.std(f1s)))

    bestidx = np.argmax(val_aucs)
    print('***Best testing AUC {:.3f}, ACC {:.3f}, PRECISION {:.3f}, RECALL {:.3f}, F1 {:.3f}'
        .format(aucs[bestidx], accs[bestidx], precisions[bestidx], recalls[bestidx], f1s[bestidx]))

    print('Saved in ', join(save_dir, 'results.csv'))