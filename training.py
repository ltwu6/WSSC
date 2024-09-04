import os
import time
import numpy as np
import argparse
import time
import random
from collections import defaultdict
from tqdm import tqdm
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("tboard/training_curve")

import model.patch_learning_dataset as patch_learning_dataset
import model.patch_learning_models as patch_learning_model
from tools import remove_store

training_category = 'all'

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

def calc_loss_l1_weighted(shape_priors, target_sdf, metrics, name):
    criterion_l1 = nn.SmoothL1Loss(reduction='none')
    loss_l1_ori = criterion_l1(shape_priors, target_sdf)
    target_mask = np.zeros(target_sdf.shape)
    target_mask[np.where(target_sdf.cpu().detach().numpy() >= 0.5)] = 1
    pred_mask = np.zeros(shape_priors.shape)
    pred_mask[np.where(shape_priors.cpu().detach().numpy() >= 0.5)] = 1
    false_pos_mask = np.zeros(shape_priors.shape)
    false_neg_mask = np.zeros(shape_priors.shape)
    false_pos_mask[np.where((pred_mask==1) & (target_mask==0))] = 1
    false_pos_mask_cuda = torch.from_numpy(false_pos_mask).cuda()
    false_neg_mask[np.where((pred_mask==0) & (target_mask==1))] = 1
    false_neg_mask_cuda = torch.from_numpy(false_neg_mask).cuda()
    loss_l1 = torch.mean(false_neg_mask_cuda * loss_l1_ori * 5 + false_pos_mask_cuda * loss_l1_ori * 3 + (1 - false_neg_mask_cuda - false_pos_mask_cuda) * loss_l1_ori) # weight false positiva / false negative
    #loss_var = torch.mean(1.0/(torch.var(shape_priors, dim=[1,2,3,4])+1e-8))
    
    # loss_l1_data = loss_l1.data.cpu().numpy()
    # metrics['loss'+name] += loss_l1_data * target_sdf.size(0)

    loss = loss_l1#+0.0001*loss_var
    loss_data = loss.data.cpu().numpy()
    metrics['loss'+name] += loss_data * target_sdf.size(0)

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

    # return loss_l1
    return loss

def print_metrics(metrics, epoch_samples, phase, epoch):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))
    save_log_path = './log_alllevel64'
    os.makedirs(save_log_path, exist_ok=True)
    with open(save_log_path+os.sep+training_category+'.txt', 'a') as wf:
        wf.write("Epoch: {} - {}: {}\n".format(epoch, phase, ", ".join(outputs)))
    return metrics['iou']/ epoch_samples

def train(device, model, dataloaders, save_epoch, step, num_epochs, output_path, lr):

    optimizer = optim.Adam(model.parameters(), lr=lr)#, weight_decay=0.0001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.5) 

    ## iterate epoches
    best_iou = 0.0
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
                if epoch%1 != 0:
                    break
                # Set model to evaluate mode
                model.eval()
            print (phase)
            # start process
            metrics = defaultdict(float)
            epoch_samples = 0
            
            pbar = tqdm(total=len(dataloaders[phase]))
            for inputs, labels, keys in dataloaders[phase]:
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)
                # zero the parameter gradients
                with torch.set_grad_enabled(phase == 'train'):
                    shape_priors_generated, _, _, _ = model(inputs)
                    loss = calc_loss_l1_weighted(shape_priors_generated, labels, metrics, '')
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                epoch_samples += inputs.size(0)
                pbar.set_postfix(loss=float(loss))
                pbar.update(1)
            pbar.close()
            iou_val = print_metrics(metrics, epoch_samples, phase, epoch)

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
            model_name = "epoch_" + str(epoch) + ".pt"
            
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
                        #required=True,
                        type=str)
    parser.add_argument("--dataset",
                        help="dataeset to be trained",
                        default="shapenet",
                        type=str)
    parser.add_argument("--train_file",
                        help="path to save the trainning dataset",
                        default="./txt_files/shapenet_train.txt",
                        type=str)
    parser.add_argument("--val_trained_file",
                        help="path to save the trained validation dataset",
                        default="./txt_files/shapenet_test.txt",
                        type=str)
    parser.add_argument("--output_path",
                        help="path to save the trained model",
                        default='',
                        type=str)
    parser.add_argument("--batch_size",
                        help="batch size for training",
                        default=1,
                        type=int)
    parser.add_argument("--num_epochs",
                        help="the number of epochs",
                        default=121,
                        type=int)
    parser.add_argument("--patch_res",
                        help="patch resoluion for patch learning stage",
                        default=4, # 32,8,4
                        type=int)
    parser.add_argument("--lr",
                        help="learning rate for training",
                        default=0.001,
                        type=float)
    parser.add_argument("--channel_num",
                        help="number of channels for learning",
                        default=128,# default:128
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
    parser.add_argument('--no_wall_aug', dest='no_wall_aug', default=False) 
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
    if os.path.exists(args.val_novel_file) is not True:
        print('Please use valid validation novel file')
        return
    if os.path.exists(args.output_path) is not True:
        os.makedirs(args.output_path,exist_ok=True)
    # create dataset
    if args.dataset == 'shapenet':
        train_set = patch_learning_dataset.ShapenetDataset(args.train_file, args.data_path)
        val_trained_set = patch_learning_dataset.ShapenetDataset(args.val_trained_file, args.data_path)
    elif args.dataset == 'scannet':
        train_set = patch_learning_dataset.ScannetDataset(args.train_file, args.data_path, use_bbox=True)
        val_trained_set = patch_learning_dataset.ScannetDataset(args.val_trained_file, args.data_path, use_bbox=True)
    else:
        print ("Please use valid datasets (shapenet, scannet)")
        return 
    print ("Finish data Loading")
    dataloaders = {
        'train': DataLoader(train_set, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker, num_workers=3),
        'val_trained': DataLoader(val_trained_set, batch_size=args.batch_size, shuffle=False, worker_init_fn=seed_worker, num_workers=3)
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

    model_refine = patch_learning_model.PSLN(args.no_batchnorm, device, args.channel_num)

    # load models if provided
    if args.load_model is not None:
        model_name = args.load_model
        model_path = os.path.join(model_name)
        model_refine.load_state_dict(torch.load(model_path))
        print ("Load model {}".format(model_name))

    model_refine = model_refine.to(device)

    # training
    print("Start training")
    start_time = time.time() 
    train(device, model_refine, dataloaders, args.save_epoch, args.step, args.num_epochs, args.output_path, args.lr)
    train_time = time.time() - start_time
    print ("training time: {}".format(train_time))

if __name__ == "__main__":
    main()
