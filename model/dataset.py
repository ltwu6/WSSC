import os
import glob
import copy
import numpy as np
import torch
from torch.utils.data import Dataset

class ShapenetDataset(Dataset):
    """
    This class reads all the training dataset for Shapenet
    """
    def __init__(self, file_name, data_path):
        """
        Initializes ShapenetDataset object. 
        """
        # path parameters
        self._file = file_name
        self._data_path = data_path
        # read data
        self._data_pairs = []
        self._mask_names = []
        self._read_data()

    def _read_data(self):
        with open(self._file, 'r') as data:
            print (self._file)
            gt_files = data.readlines()
            i = 0
            for gt_file in gt_files:
                model_path = os.path.join(self._data_path, gt_file.split('\n')[0])
                gt_file = os.path.join(model_path, "gt_voxel.npy")
                print(f'reading gt_file:{i}/{len(gt_files)}')
                i += 1
                gt = np.load(gt_file)
                for input_file in glob.glob(os.path.join(model_path, "input*_voxel.npy")):
                    inputs = np.load(input_file)
                    self._data_pairs.append([inputs, gt])
                    self._mask_names.append(input_file)
        print (len(self._data_pairs))

    def __len__(self):
        return len(self._data_pairs)

    def __getitem__(self, idx):
        name = self._mask_names[idx]
        input_sdf = copy.deepcopy(self._data_pairs[idx][0])
        gt_sdf = copy.deepcopy(self._data_pairs[idx][1])
        # get final data
        input_sdf = torch.from_numpy(input_sdf).unsqueeze(0).float()
        gt_sdf = torch.from_numpy(gt_sdf).unsqueeze(0).float()
        return [input_sdf, gt_sdf, name]

class ScannetDataset(Dataset):
    """
    This class reads all the training dataset for Scannet
    """
    def __init__(self, file_name, data_path):
        """
        Initializes ScannetDataset object. 
        """
        # path parameters
        self._file = file_name
        self._data_path = data_path
        self._input_sdf = []
        self._gt_sdf = []
        self._mask_names = []
        self._read_data()

    def _read_data(self):
        with open(self._file, 'r') as data:
            print (self._file)
            mask_files = data.readlines()
            for mask_file in mask_files:
                mask_file = os.path.join(self._data_path, mask_file.split('\n')[0])
                sdf_file = mask_file[:-4] + "_input_voxel.npy"
                gt_sdf_file = mask_file[:-8] + "scaled_gt_voxel.npy"
                if os.path.exists(sdf_file) is False or os.path.exists(gt_sdf_file) is False:
                    continue
                input_sdf = np.load(sdf_file)
                
                self._input_sdf.append(input_sdf)
                gt_sdf = np.load(gt_sdf_file)
                self._gt_sdf.append(gt_sdf)
                self._mask_names.append(mask_file)
        print (len(self._mask_names))

    def __len__(self):
        """
        This function returns the number of traing samples
        """
        return len(self._mask_names)

    def __getitem__(self, idx):
        input_sdf = copy.deepcopy(self._input_sdf[idx])
        gt_sdf = copy.deepcopy(self._gt_sdf[idx])
        input_sdf = torch.from_numpy(input_sdf).unsqueeze(0).float()
        gt_sdf = torch.from_numpy(gt_sdf).unsqueeze(0)
        return [input_sdf, gt_sdf, self._mask_names[idx]]

class ShapenetDataset_CASR_Train(Dataset):
    """
    This class reads all the training dataset for Shapenet
    """
    def __init__(self, file_name, data_path, category='all', coarse_path=''):
        """
        Initializes ShapenetDataset for CaSR.
        """
        # path parameters
        self._file = file_name
        self._data_path = data_path
        self._coarse_path = coarse_path
        # read data
        self._data_pairs = []
        self._mask_names = []
        self._ms_parts = []
        self._ms_masks = []
        self._pnum = []
        self.ctg = category
        self._read_data()

    def _read_data(self):
        with open(self._file, 'r') as data:
            print (self._file)
            gt_files = data.readlines()    
            for gt_file in gt_files:
                if gt_file.split('/')[0] != self.ctg and self.ctg != 'all':
                    continue
                c_path = os.path.join(self._coarse_path, gt_file.strip())
                model_path = os.path.join(self._data_path, gt_file.split('\n')[0])
                gt_file = os.path.join(model_path, "gt_voxel.npy")
                pp1 = np.load(os.path.join(model_path, 'input_4_voxel.npy')).astype(np.int32)
                pp2 = np.load(os.path.join(model_path, 'input_5_voxel.npy')).astype(np.int32)
                pp3 = np.load(os.path.join(model_path, 'input_6_voxel.npy')).astype(np.int32)
                pp4 = np.load(os.path.join(model_path, 'input_7_voxel.npy')).astype(np.int32)
                fusion_part = pp1|pp2|pp3|pp4
                for input_file in glob.glob(os.path.join(c_path, "input*_pred.npz")):
                    part_input_path = os.path.join(model_path, 'input_'+input_file.split('/')[-1].split('_')[1]+'_voxel.npy')
                    part_input = np.load(part_input_path)
                    with np.load(input_file, 'rb') as data:
                        cinputs = data['predicted_voxels']
                        point_num = min(np.sum(part_input), 32768/2.5, np.sum(fusion_part)/2.5)
                        point_num = max(point_num, np.sum(fusion_part)*0.52)
                        cinputs[np.where(cinputs>=0.5)] = 1
                        cinputs[np.where(cinputs<0.5)] = 0
                        msparts = (1-fusion_part)*cinputs
                        self._data_pairs.append([cinputs, pp1, pp2, pp3, pp4])
                        self._mask_names.append(input_file)
                        self._pnum.append(point_num)
                        self._ms_parts.append(msparts)
        print (len(self._data_pairs))

    def __len__(self):
        return len(self._data_pairs)

    def __getitem__(self, idx):
        name = self._mask_names[idx]
        input_sdf = copy.deepcopy(self._data_pairs[idx][0])
        gt_sdf1 = copy.deepcopy(self._data_pairs[idx][1])
        gt_sdf2 = copy.deepcopy(self._data_pairs[idx][2])
        gt_sdf3 = copy.deepcopy(self._data_pairs[idx][3])
        gt_sdf4 = copy.deepcopy(self._data_pairs[idx][4])

        m_parts = torch.from_numpy(self._ms_parts[idx]).unsqueeze(0).float()
        input_sdf = torch.from_numpy(input_sdf).unsqueeze(0).float()
        gt_sdf1 = torch.from_numpy(gt_sdf1).unsqueeze(0).float()
        gt_sdf2 = torch.from_numpy(gt_sdf2).unsqueeze(0).float()
        gt_sdf3 = torch.from_numpy(gt_sdf3).unsqueeze(0).float()
        gt_sdf4 = torch.from_numpy(gt_sdf4).unsqueeze(0).float()
        pnum = self._pnum[idx]
        return [input_sdf, gt_sdf1, gt_sdf2,gt_sdf3, gt_sdf4, m_parts, name, pnum]

class ShapenetDataset_CASR_Test(Dataset):
    """
    This class reads all the training dataset for Shapenet
    """
    def __init__(self, file_name, data_path, category='all', coarse_path=''):
        """
        Initializes ShapenetDataset For CaSR.
        """
        # path parameters
        self._file = file_name
        self._data_path = data_path
        self._coarse_path = coarse_path
        # read data
        self._data_pairs = []
        self._mask_names = []
        self._ms_parts = []
        self._pnum = []
        self.ctg = category
        self._read_data()

    def _read_data(self):
        with open(self._file, 'r') as data:
            print (self._file)
            gt_files = data.readlines()
            for gt_file in gt_files:
                if gt_file.split('/')[0] != self.ctg and self.ctg != 'all':
                    continue
                c_path = os.path.join(self._coarse_path, gt_file.strip())
                model_path = os.path.join(self._data_path, gt_file.split('\n')[0])
                gt_file = os.path.join(model_path, "gt_voxel.npy")
                gt = np.load(gt_file)
                for input_file in glob.glob(os.path.join(c_path, "input*_pred.npz")):
                    with np.load(input_file, 'rb') as data:
                        cinputs = data['predicted_voxels']
                        cinputs[np.where(cinputs>=0.5)] = 1
                        cinputs[np.where(cinputs<0.5)] = 0
                        self._data_pairs.append([cinputs, gt])
                        self._mask_names.append(input_file)
        print (len(self._data_pairs))

    def __len__(self):
        return len(self._data_pairs)

    def __getitem__(self, idx):
        name = self._mask_names[idx]
        input_sdf = copy.deepcopy(self._data_pairs[idx][0])
        gt_sdf = copy.deepcopy(self._data_pairs[idx][1])
        input_sdf = torch.from_numpy(input_sdf).unsqueeze(0).float()
        gt_sdf = torch.from_numpy(gt_sdf).unsqueeze(0).float()
        return [input_sdf, gt_sdf, name]

















