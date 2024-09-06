# WSSC
This is the official implementation for the paper  [3D Shape Completion on Unseen Categories: A Weakly-supervised Approach](https://arxiv.org/pdf/2401.10578.pdf)

![image](https://github.com/ltwu6/WSSC/blob/main/model/flowchart.png)

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
  journal={arXiv preprint arXiv:2401.10578},
  year={2024}
}
```
