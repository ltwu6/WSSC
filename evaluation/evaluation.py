import numpy as np
import os
import glob
import time
import argparse
import copy
import torch
from chamfer_distance import ChamferDistance
import em_distance.emd_module as emddst
emd_metric = emddst.emdModule()
cat_map_inv = {
            '02773838':'bag',
            '03636649':'lamp',
            '02808440':'bathtub',
            '02818832':'bed',
            '02801938':'basket',
            '04004475':'printer',
            '03642806':'laptop',
            '02828884':'bench',
            'all':'all'
        }

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

class Evaluation:
    """
    This class caluate Chamfer Distance and IOU between GTs and predictions

    Steps:
    1. Read GT and prediction
    2. Convert voxels to meshes
    3. Sample points on the meshes and then calcualte chamfer distance
    4. Calculate IOU and F1
    """
    def __init__(self, args):
        """
        Initializes Evaluation object. Sets up related pathes and other parameters.

        :param args: arguments from the commandline
        :type args: argparse.Namespace
        """
        # init path for related files
        self._root = args.root
        if os.path.exists(self._root) is not True:
            print ('Please enter valid data root path')
            return
        self._pred_path = args.pred_path
        if os.path.exists(self._pred_path) is not True:
            print('Please enter valid pred path')
            return
        self._test_file = args.test_file
        if os.path.exists(self._test_file) is not True:
            print('Please enter valid test file path')
            return
        
        # set paths based on dataset
        self._data_path = os.path.join(self._root)
        self._pred_path = os.path.join(self._pred_path)
        # get output file
        self._points_n = args.points_n
        self._chamfer_dist = ChamferDistance()

    def calculate_iou(self, gt, pred, threshold):
        bool_true_voxels = gt > threshold
        bool_pred_voxels = pred > threshold
        total_union = (bool_true_voxels | bool_pred_voxels).sum()
        total_intersection = (bool_true_voxels & bool_pred_voxels).sum()
        return (total_intersection / total_union)

    def calculate_f1(self, gt_mask, pred_mask):
        mask_t = copy.deepcopy(gt_mask)
        mask_t[np.where(pred_mask==1)] += 10
        miss = len(np.where(mask_t==1)[0]) / np.sum(gt_mask)
        redundant = len(np.where(mask_t==10)[0]) / np.sum(gt_mask)
        f1 = np.sum(np.logical_and(gt_mask, pred_mask)) / (np.sum(np.logical_and(gt_mask, pred_mask)) + 0.5 * np.sum(np.logical_xor(gt_mask, pred_mask)))  
        return miss, redundant, f1

    def calculate_cd(self, gt_mask, pred_mask,dst):
        # get points for prediction
        if np.sum(pred_mask) == 0:
            pred_points = np.zeros((self._points_n, 3))
        else:
            pred_points = self.get_voxel_points(pred_mask,32)
        gt_points = self.get_voxel_points(gt_mask,32)

        # calcualte CD
        gt_points_torch = torch.from_numpy(gt_points).cuda().unsqueeze(0).float() 
        pred_points_torch = torch.from_numpy(pred_points).cuda().unsqueeze(0).float()
        dist1, dist2 = self._chamfer_dist(gt_points_torch, pred_points_torch)
        eps = 1e-10
        loss = torch.sqrt(dist1 + eps).mean(1) + torch.sqrt(dist2 + eps).mean(1)
        return loss.detach().cpu().numpy()*100

    def get_voxel_points(self, V,voxel_res):
        indices = np.argwhere(V == 1)
        pc = indices.astype(float) / voxel_res
        pc = pc-0.5
        return pc
    def evaluate_shapenet(self):
        """
        This function gets final cd evaluation results
        """
        start_time = time.time()
        eval_res = {}
        # iterate all the files
        with open(self._test_file, 'r') as data:
            gt_files = data.readlines()
            for gt_file in gt_files:
                gt_file = gt_file.split('\n')[0]
                pred_model_path = os.path.join(self._pred_path, gt_file)
                if not os.path.exists(pred_model_path):
                    continue
                gt_model_path = os.path.join(self._data_path, gt_file)
                gt_name = os.path.join(gt_model_path, "gt_voxel.npy") 
                gt = np.load(gt_name)
                gt_mask = gt
                for pred_name in glob.glob(os.path.join(pred_model_path, "*_pred.npz")): 
                    with np.load(pred_name, 'rb') as data:
                        pred = data['predicted_voxels'].squeeze()
                        pred_mask = np.zeros(pred.shape)
                        pred_mask[np.where(pred>=0.5)] = 1  
                    # evaluate IOU, f1 and cd
                    iou = self.calculate_iou(gt_mask, pred_mask, 0.5)
                    miss, redundant, f1 = self.calculate_f1(gt_mask, pred_mask)
                    save_path = None
                    cd = self.calculate_cd(gt_mask, pred_mask,save_path)
                    eval_res[pred_name] = [cd[0], iou, miss, redundant, f1, pred_name]
                
        print ("costing time:")
        print (time.time() - start_time)
        return eval_res

    def evaluate_scannet(self):
        """
        This function gets final cd evaluation results
        """
        start_time = time.time()
        eval_res = {}
        with open(self._test_file, 'r') as data:
            gt_files = os.listdir(self._pred_path)
            for gt_file in gt_files:
                mask_file = os.path.join(self._data_path, self._pred_path.split('/')[-1], gt_file)
                pred_file = os.path.join(self._pred_path,gt_file)
                gt_sdf_file = mask_file[:-4].replace('mask_pred','scaled_gt_voxel') + ".npy"
                if os.path.exists(gt_sdf_file) is False:
                    raise Exception('gt_sdf_file path error: ',gt_sdf_file)
                if os.path.exists(pred_file) is False:
                    raise Exception('pred_file path error: ',pred_file)
                with open(gt_sdf_file, 'rb') as data:
                    gt = np.load(data)
                    gt_mask = gt
                with np.load(pred_file, 'rb') as data:
                    pred = data['predicted_voxels'][0]
                    pred_mask = np.zeros(pred.shape)
                    pred_mask[np.where(pred>=0.5)] = 1
                iou = self.calculate_iou(gt_mask, pred_mask, 0.5)
                miss, redundant, f1 = self.calculate_f1(gt_mask, pred_mask)
                cd = self.calculate_cd(gt_mask, pred_mask)
                eval_res[self._pred_path.split('/')[-1]+os.sep+gt_file] = [cd[0], iou, miss, redundant, f1, gt_file]
        
        print ("costing time:")
        print (time.time() - start_time)
        return eval_res
    
    def evaluate_scannet_category(self):
        """
        This function gets final cd evaluation results
        """
        start_time = time.time()
        eval_res = {}
        # iterate all the files
        point_num = []
        cnt = 0
        with open(self._test_file, 'r') as data:
            gt_files = os.listdir(self._pred_path)
            for gt_file in gt_files:
                cnt += 1
                mask_file = os.path.join(self._data_path, self._pred_path.split('/')[-1], gt_file)
                pred_file = os.path.join(self._pred_path,gt_file)
                gt_sdf_file = mask_file[:-9].replace('mask_pred','scaled_gt_voxel') + ".npy"
                if os.path.exists(gt_sdf_file) is False:
                    raise Exception('gt_sdf_file path error: ',gt_sdf_file)
                if os.path.exists(pred_file) is False:
                    raise Exception('pred_file path error: ',pred_file)
                with open(gt_sdf_file, 'rb') as data:
                    gt = np.load(data)
                    gt_mask = gt
                    point_num.append(np.sum(gt_mask))
                with np.load(pred_file, 'rb') as data:
                    pred = data['predicted_voxels']
                    pred_mask = np.zeros(pred.shape)
                    pred_mask[np.where(pred>=0.5)] = 1

                iou = self.calculate_iou(gt_mask, pred_mask, 0.5)
                miss, redundant, f1 = self.calculate_f1(gt_mask, pred_mask)
                cd = self.calculate_cd(gt_mask, pred_mask)
                eval_res[self._pred_path.split('/')[-1]+os.sep+gt_file] = [cd[0], iou, miss, redundant, f1, gt_file]
        avg_pnum = np.sum(point_num)/cnt
        print('avg pnum: ', avg_pnum)
        print ("costing time:")
        print (time.time() - start_time)
        return eval_res


def parse_arguments():
    """
    Generates a command line parser that supports all arguments used by the
    tool.

    :return: Parser with all arguments.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    # required parameters
    parser.add_argument("--dataset",
                        help="dataeset to be trained",
                        default="shapenet",
                        type=str)
    parser.add_argument("--root",
                        help="root for data",
                        default='',
                        type=str)
    parser.add_argument("--pred_path",
                        help="path for predicted results",
                        default='',
                        type=str)
    # optional parameters
    parser.add_argument("--test_file",
                        help="test samples names",
                        default="../txt_files/shapenet_test.txt",
                        type=str)
    parser.add_argument("--points_n",
                        help="the number of sampling points for chamfer distance",
                        default=10240,
                        type=int)
    return parser

def print_results(eval_res, dataset):
    cats_iou = {}
    if dataset == 'shapenet':
        for name, eval_r in eval_res.items():
            cat = name.split('/')[-3]
            if cat in cats_iou:
                cats_iou[cat].append(eval_r)
            else:
                cats_iou[cat] = [eval_r]
    if dataset == 'scannet':
        for name, eval_r in eval_res.items():
            cat = name.split('/')[0]
            if cat in cats_iou:
                cats_iou[cat].append(eval_r)
            else:
                cats_iou[cat] = [eval_r]
    sum_cd = 0
    sum_iou = 0
    sum_miss = 0
    sum_red = 0
    sum_f1 = 0
    sum_len = 0
    cat_cds = []
    cat_ious = []
    cat_misses = []
    cat_reds = []
    cat_f1s = []
    good = []
    bad = []
    cat_names = []
    for cat, data_l in cats_iou.items():
        print (cat)
        print (len(data_l))
        cat_names.append(cat)
        cat_cd = []
        cat_iou = []
        cat_miss = []
        cat_red = []
        cat_f1 = []
        names = []
        for cd, iou, miss, red, f1, name in data_l:
            sum_cd += cd
            sum_iou += iou
            sum_miss += miss
            sum_red += red
            sum_f1 += f1
            cat_cd.append(cd)
            cat_iou.append(iou)
            cat_miss.append(miss)
            cat_red.append(red)
            cat_f1.append(f1)
            names.append(name)
        cat_cds.append(np.array(cat_cd).mean())    
        cat_ious.append(np.array(cat_iou).mean())
        cat_misses.append(np.array(cat_miss).mean())
        cat_reds.append(np.array(cat_red).mean())
        cat_f1s.append(np.array(cat_f1).mean())
        sum_len += len(data_l)
        bad_idx = cat_iou.index(min(cat_iou))
        bad.append(names[bad_idx])
        good_idx = cat_iou.index(max(cat_iou))
        good.append(names[good_idx])
    print('CD of each category--------------------------')
    for ci,cd in enumerate(cat_cds):
        print (cat_names[ci],':',np.round(cd,5),'--', cat_map_inv[cat_names[ci]])
    print('IOU of each category--------------------------')
    for ii,iou in enumerate(cat_ious):
        print (cat_names[ii],':',np.round(iou,5),'--', cat_map_inv[cat_names[ii]])
    print('F1 of each category--------------------------')
    for fi,f1 in enumerate(cat_f1s):
        print (cat_names[fi],':',np.round(f1,5),'--', cat_map_inv[cat_names[fi]])
    print ("instance_cd:  ",np.round(sum_cd / sum_len, 5), '|', "category cd:  ", np.round(np.array(cat_cds).mean(),5))
    print ("instance_iou: ", np.round(sum_iou / sum_len, 5), '|', "category iou: ", np.round(np.array(cat_ious).mean(), 5))
    print ("instance_f1:  ", np.round(sum_f1 / sum_len, 5), '|', "category f1:  ", np.round(np.array(cat_f1s).mean(), 5))
    print (sum_len)
    print ('good')

def main():
    """
    Evaluate chamfer distance on prediction
    """
    args = parse_arguments().parse_args()
    evaluator = Evaluation(args)
    if args.dataset == "shapenet":
        eval_res = evaluator.evaluate_shapenet()
        print_results(eval_res, "shapenet")
    if args.dataset == "scannet":
        eval_res = evaluator.evaluate_scannet_category()
        print_results(eval_res, "scannet")
    print(args.pred_path)

if __name__ == "__main__":
    main()
