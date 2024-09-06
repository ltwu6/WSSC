import os
import time
import numpy as np
import argparse
import time
import itertools
import random
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("tboard/training_curve")

import model.dataset as data_load
import model.networks as networks

training_category = 'lamp'
cat_map = {
            'bag':'02773838',
            'lamp':'03636649',
            'bathtub':'02808440',
            'bed':'02818832',
            'basket':'02801938',
            'printer':'04004475',
            'laptop':'03642806',
            'bench':'02828884',
            'all':'all'
        }
cat_map_scannet = {
            'bag':'02773838',
            'lamp':'03636649',
            'bathtub':'02808440',
            'bed':'02818832',
            'basket':'02801938',
            'printer':'04004475',
            'all':'all'
        }

def calc_loss_l1_part(shape_priors, target_sdf, point_num, pnum_flag, weight_flag,weight_map, metrics, name, device, pweight=2.5):
    criterion_l1 = nn.SmoothL1Loss(reduction='none')
    criterion_l2 = nn.MSELoss(reduction='none')
    loss_l1_ori = criterion_l1(shape_priors, target_sdf)
    if weight_flag:
        loss_l1 = torch.mean(loss_l1_ori * weight_map) # weight false positiva / false negative
    else:
        loss_l1 = torch.mean(loss_l1_ori) # weight false positiva / false negative
    predict_num = torch.sum(shape_priors, dim=[1,2,3,4])
    
    loss_pnum = torch.mean(criterion_l1(torch.ones(predict_num.shape,device=device)*point_num*pweight,predict_num))
    loss_var = torch.mean(1.0/(torch.var(shape_priors, dim=[1,2,3,4])+1e-8))

    if pnum_flag:
        loss_total = loss_l1+0.00001*loss_pnum+0.0001*loss_var##+0.01*loss_rep
    else:
        loss_total = loss_l1
    loss_total_data = loss_total.data.cpu().numpy()
    metrics['loss'+name] += loss_total_data * target_sdf.size(0)

    iou_sum = 0
    for idx, p in enumerate(shape_priors):
        # calculate IOU
        new_p = np.zeros(p[0].shape)
        new_p[np.where(p[0].cpu().detach().numpy() >= 0.5)] = 1
        new_mask = np.zeros(target_sdf[idx][0].shape)
        new_mask[np.where(target_sdf[idx][0].cpu().detach().numpy() >= 0.5)] = 1
        result = new_p + new_mask
        iou_sum += (np.sum(np.array(result) == 2) / np.sum(np.array(result) >= 1))
    metrics['iou'+name] += iou_sum

    return loss_total

def calc_loss_l1_coares_part(shape_priors, target_sdf, weight_flag,weight_map):
    criterion_l1 = nn.SmoothL1Loss(reduction='none')
    loss_l1_ori = criterion_l1(shape_priors, target_sdf)
    
    if weight_flag:
        loss_l1 = torch.mean(loss_l1_ori * weight_map) # weight false positiva / false negative
    else:
        loss_l1 = torch.mean(loss_l1_ori) # weight false positiva / false negative
    loss_total = loss_l1
    return loss_total

def calc_loss_l1_part_val(shape_priors, target_sdf, metrics, name):
    criterion_l1 = nn.SmoothL1Loss(reduction='none')
    loss_l1_ori = criterion_l1(shape_priors, target_sdf)
    
    loss_l1 = torch.mean(loss_l1_ori) # weight false positiva / false negative
    
    loss_total = loss_l1
    loss_total_data = loss_total.data.cpu().numpy()
    metrics['loss'+name] += loss_total_data * target_sdf.size(0)

    iou_sum = 0
    for idx, p in enumerate(shape_priors):
        # calculate IOU
        new_p = np.zeros(p[0].shape)
        new_p[np.where(p[0].cpu().detach().numpy() >= 0.5)] = 1
        new_mask = np.zeros(target_sdf[idx][0].shape)
        new_mask[np.where(target_sdf[idx][0].cpu().detach().numpy() >= 0.5)] = 1
        result = new_p + new_mask
        iou_sum += (np.sum(np.array(result) == 2) / np.sum(np.array(result) >= 1))
    metrics['iou'+name] += iou_sum

    return loss_total

def print_metrics(metrics, epoch_samples, phase,epoch):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))
    os.makedirs(save_log_path, exist_ok=True)
    with open(save_log_path+os.sep+training_category+'.txt', 'a') as wf:
        wf.write("Epoch: {} - {}: {}\n".format(epoch, phase, ", ".join(outputs)))    
    return metrics['iou']/ epoch_samples

def train_multiview(device, model, dataloaders, save_epoch, step, num_epochs, output_path, lr):
    """
    This function trains a model based on the training and validation datasets
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)#, weight_decay=0.0001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.5)

    pointnum_weight=2.5
    coarsegt_weight = 0.5

    for epoch in range(num_epochs):
        print(f'Training: epoch {str(epoch)}/{str(num_epochs)}')
        torch.cuda.empty_cache()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        since = time.time()
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val_trained']:
            if phase == 'train':
                # Set model to training mode
                model.train()
            else:
                if epoch%5 != 0:
                    break
                # Set model to evaluate mode
                model.eval()
            # start process
            metrics = defaultdict(float)
            epoch_samples = 0
            
            pbar = tqdm(total=len(dataloaders[phase]))
            if phase == 'train':
                for inputs, labels1, labels2,labels3,labels4, mpart, keys, pnum in dataloaders[phase]:
                    inputs = inputs.float().to(device)
                    labels1 = labels1.float().to(device)
                    labels2 = labels2.float().to(device)
                    labels3 = labels3.float().to(device)
                    labels4 = labels4.float().to(device)
                    mpart = mpart.float().to(device)
                    pnum = pnum.float().to(device)
                    with torch.set_grad_enabled(phase == 'train'):
                        shape_priors_generated= model(inputs)
                        if phase == 'train':
                            weight_flag = True
                            weight_mask1 = labels1.clone().detach()
                            weight_mask1 = weight_mask1.float().to(device)
                            weight_mask2 = labels2.clone().detach()
                            weight_mask2 = weight_mask2.float().to(device)
                            weight_mask3 = labels3.clone().detach()
                            weight_mask3 = weight_mask3.float().to(device)
                            weight_mask4 = labels4.clone().detach()
                            weight_mask4 = weight_mask4.float().to(device)
                            coarse_gt = inputs.clone().detach()
                            weight_mask_coarse = mpart.clone().detach()
                            weight_mask_coarse = weight_mask_coarse.float().to(device)
                        
                            loss1 = calc_loss_l1_part(shape_priors_generated, labels1, pnum,pnum_flag=True, weight_flag=weight_flag,weight_map=weight_mask1, metrics=metrics, name='', device=device,pweight=pointnum_weight)
                            loss2 = calc_loss_l1_part(shape_priors_generated, labels2, pnum,pnum_flag=False, weight_flag=weight_flag,weight_map=weight_mask2, metrics=metrics, name='', device=device)
                            loss3 = calc_loss_l1_part(shape_priors_generated, labels3, pnum,pnum_flag=False, weight_flag=weight_flag,weight_map=weight_mask3, metrics=metrics, name='', device=device)
                            loss4 = calc_loss_l1_part(shape_priors_generated, labels4, pnum,pnum_flag=False, weight_flag=weight_flag,weight_map=weight_mask4, metrics=metrics, name='', device=device)
                            loss_cp = calc_loss_l1_coares_part(shape_priors_generated, coarse_gt, weight_flag=weight_flag, weight_map=weight_mask_coarse)
                            loss = loss1+loss2+loss3+loss4+coarsegt_weight*loss_cp
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            
                    epoch_samples += inputs.size(0)
                    pbar.set_postfix(loss=float(loss))
                    pbar.update(1)
            elif phase == 'val_trained':
                for inputs, labels, keys in dataloaders[phase]:
                    inputs = inputs.float().to(device)
                    labels = labels.float().to(device)
                    weight_flag = False
                    shape_priors_generated= model(inputs)
                    loss = calc_loss_l1_part_val(shape_priors_generated, labels, metrics=metrics, name='')
                    pbar.set_postfix(loss=float(loss))
                    pbar.update(1)
            pbar.close()
            epoch_loss = metrics['loss'] / epoch_samples
            epoch_iou = metrics['iou'] / epoch_samples
            
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                writer.add_scalar("Loss/train", epoch_loss, epoch)
                writer.add_scalar("iou/train", epoch_iou, epoch)
            elif phase == 'val_trained':
                writer.add_scalar("Loss/val_trained", epoch_loss, epoch)
                writer.add_scalar("iou/val_trained", epoch_iou, epoch)
        
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        exp_lr_scheduler.step()
        # save model
        if (epoch + 1) % save_epoch == 0:
            model_name =  "CaSR_epoch_" + str(epoch) + ".pt"
            model_path = os.path.join(output_path, model_name)
            print ("save model {}".format(model_name))
            torch.save(model.state_dict(), model_path)
    return

def parse_arguments():
    """
    Generates a command line parser that supports all arguments used by the
    tool.

    :return: Parser with all arguments.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    # optimal parameters
    parser.add_argument("--data_path",
                        help="path for getting data",
                        default="",
                        type=str)
    parser.add_argument("--dataset",
                        help="dataeset to be trained",
                        default="shapenet",
                        type=str)
    parser.add_argument("--train_file",
                        help="path to save the trainning dataset",
                        default="./txt_files/shapenet_test.txt",
                        type=str)
    parser.add_argument("--val_trained_file",
                        help="path to save the trained validation dataset",
                        default="./txt_files/shapenet_test.txt",
                        type=str)
    parser.add_argument("--coarse_path",
                        help="path of coarse results predicted by CoSL",
                        default="",
                        type=str)
    parser.add_argument("--output_path",
                        help="path to save the trained model",
                        default='',
                        type=str)
    parser.add_argument("--batch_size",
                        help="batch size for training",
                        default=10,
                        type=int)
    parser.add_argument("--num_epochs",
                        help="the number of epochs",
                        default=120,
                        type=int)

    parser.add_argument("--lr",
                        help="learning rate for training",
                        default=0.0001,
                        type=float)
    parser.add_argument("--channel_num",
                        help="number of channels for learning",
                        default=128,
                        type=int)
    parser.add_argument("--save_epoch",
                        help="save model after how many epochs",
                        default=5,
                        type=int)
    parser.add_argument("--step",
                        help="decrease lr after how may epochs",
                        default=50,
                        type=int)
    parser.add_argument("--load_model",
                        default=None,
                        help="the model for continue training",
                        type=str)
    parser.add_argument("--gpu",
                        help="gpu index for training",
                        default=0,
                        type=int)
    parser.add_argument('--no_batchnorm', dest='no_batchnorm', action='store_false')
    parser.set_defaults(feature=False)

    return parser

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    """
    generates binary masks for training
    """
    args = parse_arguments().parse_args()
    # get file path
    if os.path.exists(args.data_path) is not True:
        print('Please use valid data_path')
        return
    if os.path.exists(args.train_file) is not True:
        print('Please use valid train file')
        return
    if os.path.exists(args.val_trained_file) is not True:
        print('Please use valid validation trained file')
        return
    if os.path.exists(args.output_path) is not True:
        os.makedirs(args.output_path,exist_ok=True)
    # create dataset
    train_set = data_load.ShapenetDataset_CASR_Train(args.train_file, args.data_path, category=cat_map[training_category],coarse_path=args.coarse_path)
    val_trained_set = data_load.ShapenetDataset_CASR_Test(args.train_file, args.data_path, category=cat_map[training_category],coarse_path=args.coarse_path)
    print ("Finish data Loading")
    dataloaders = {
        'train': DataLoader(train_set, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker, num_workers=3),
        'val_trained': DataLoader(val_trained_set, batch_size=args.batch_size, shuffle=False, worker_init_fn=seed_worker, num_workers=3),
    }

    # set up device and model
    print("Init model")
    gpu = 'cuda:' + str(args.gpu)
    print ("using gpu: " + gpu)
    device = torch.device(gpu if torch.cuda.is_available() else 'cpu')
    print (device)
    manual_seed = 1
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    model_refine = networks.PSLN_CaSR(args.no_batchnorm, device, args.channel_num, ctg=cat_map[training_category])

    # load models if provided
    if args.load_model is not None:
        model_name = args.load_model
        model_path = model_name
        model_refine.load_state_dict(torch.load(model_path))
        print ("Load model {}".format(model_name))

    model_refine = model_refine.to(device)
    
    # training
    print("Start training")
    start_time = time.time() 
    train_multiview(device, model_refine, dataloaders, args.save_epoch, args.step, args.num_epochs, args.output_path, args.lr)
    train_time = time.time() - start_time
    print ("training time: {}".format(train_time))

if __name__ == "__main__":
    main()
