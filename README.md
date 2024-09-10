# WSSC
This is the official implementation for the paper  [3D Shape Completion on Unseen Categories: A Weakly-supervised Approach](https://arxiv.org/pdf/2401.10578.pdf)(Accepted by IEEE TVCG)

![image](https://github.com/ltwu6/WSSC/blob/main/model/flowchart.png)

### Abstract
3D shapes captured by scanning devices are often incomplete due to occlusion.  3D shape completion methods have been explored to tackle this limitation. However, most of these methods are only trained and tested on a subset of categories, resulting in poor generalization to unseen categories. In this paper, we propose a novel weakly-supervised framework to reconstruct the complete shapes from unseen categories. We first propose an end-to-end prior-assisted shape learning network that leverages data from the seen categories to infer a coarse shape. Specifically, we construct a prior bank consisting of representative shapes from the seen categories. Then, we design a multi-scale pattern correlation module for learning the complete shape of the input by analyzing the correlation between local patterns within the input and the priors at various scales. In addition, we propose a self-supervised shape refinement model to further refine the coarse shape. Considering the shape variability of 3D objects across categories, we construct a category-specific prior bank to facilitate shape refinement. Then, we devise a voxel-based partial matching loss and leverage the partial scans to drive the refinement process. Extensive experimental results show that our approach is superior to state-of-the-art methods by a large margin.

## Environment
Python: 3.9  
PyTorch: 0.10.1  
Cuda: 11.1  

## Dataset
[ShapeNet and Scannet](https://github.com/yuchenrao/PatchComplete)

## Get Started
### Training
For CoSL stage:
```
CUDA_VISIBLE_DEVICES=[gpu_id] python train.py 
```
For CaSR stage:
```
CUDA_VISIBLE_DEVICES=[gpu_id] python train_refine.py --
```
### Evaluation
```
cd evaluation
CUDA_VISIBLE_DEVICES=[gpu_id] python evaluation.py
```

## Acknowledgements
Some codes are borrowed from [PatchComplete](https://github.com/yuchenrao/PatchComplete). Thanks for their great work.

## Cite this work
```
@article{wu20243d,
  title={3D Shape Completion on Unseen Categories: A Weakly-supervised Approach},
  author={Wu, Lintai and Hou, Junhui and Song, Linqi and Xu, Yong},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2024}
}
```
