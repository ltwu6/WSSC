import numpy as np
import os
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MRFusion_Att(nn.Module):
    def __init__(self, in_features, out_features, kernel, stride, padding, r=8, L=16):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(MRFusion_Att, self).__init__()
        d = max(int(in_features / r), L)
        self.M = len(kernel)
        self.in_features = in_features
        self.out_features = out_features
        self.convs = nn.ModuleList([])
        for i in range(self.M):
            self.convs.append(nn.Sequential(
                nn.Conv3d(in_features, out_features, kernel_size=kernel[i], stride=stride[i], padding=padding[i]),
                nn.BatchNorm3d(out_features),
                nn.ReLU(inplace=False)
            ))
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(out_features, d)
        self.fcs = nn.ModuleList([])
        self.share_conv = nn.Conv2d(d, out_features, 1, 1)
        for i in range(self.M):
            self.fcs.append(
                nn.Linear(out_features, d)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x)  # E, b c d h w
            # print('shape of fea after conv: ', fea.shape)#[b c d h w]
            fea_p = self.gap(fea)  # p
            # print('shape of fea after gap: ', fea_p.shape)# [b c 1 1 1]
            fea_p = fea_p.view(-1, self.out_features)
            fea_d = self.fcs[i](fea_p).unsqueeze(-1).unsqueeze(-1)  # d: bx16x1x1
            # print('shape of fea after fc: ', fea_d.shape)#[b c1 1 1]
            if i == 0:
                feas = fea.unsqueeze(dim=1)  # b 1 c d h w
                vectors = fea_d
            else:
                feas = torch.cat([feas, fea.unsqueeze(dim=1)], dim=1)  # b 2 c d h w
                # print('shape of feas: ', feas.shape) # [b 2 c d h w]
                #print('feas[0][0]: ', feas[0][0])
                #print('feas[0][1]: ', feas[0][1])
                vectors = torch.cat([vectors, fea_d], dim=2)  # bxcx2x1
        att_vectors = self.share_conv(vectors)  # bx64x2x1
        # print('shape of attvec: ', att_vectors.shape) # [b c 2 1]
        #print('attvec[0]: ', att_vectors[0])
        att_vectors = self.sigmoid(att_vectors)  # bx64x2x1
        # print('shape of attvec after sigmoid: ', att_vectors.shape)
        #print('attvec[0] after sigmoid: ', att_vectors[0])
        for j, vec in enumerate(att_vectors.chunk(2,2)):  # vec: bx64x1x1
            vec_sg = vec.unsqueeze(dim=1)  # bx1x64x1x1
            # print('shape of vec_sg: ', vec_sg.shape)#[b 1 c 1 1]
            if j == 0:
                vecs = vec_sg
                #print('vec_sg[0][0]: ', vec_sg[0][0])
            else:
                vecs = torch.cat([vecs, vec_sg], dim=1).unsqueeze(-1)  # bx2x64x1x1
                # print('shape of vecs: ', vecs.shape)#[b 2 c 1 1 1]
                #print('vecs[0][1]: ', vecs[0][1])
        fea_v = (feas * vecs).sum(dim=1)  # b c d h w
        # print('shape of fea_v: ', fea_v.shape) # [b c d h w]
        #print('fea_v[0]: ', fea_v[0])
        return fea_v

class MRFusion(nn.Module):
    def __init__(self, in_features, out_features, kernel, stride, padding):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(MRFusion, self).__init__()
        self.M = len(kernel)
        self.in_features = in_features
        self.out_features = out_features
        self.convs = nn.ModuleList([])
        for i in range(self.M):
            self.convs.append(nn.Sequential(
                nn.Conv3d(in_features, out_features, kernel_size=kernel[i], stride=stride[i], padding=padding[i]),
                nn.BatchNorm3d(out_features),
                nn.ReLU(inplace=False)
            ))

    def forward(self, x):
        feat_list = []
        for i, conv in enumerate(self.convs):
            fea = conv(x)  # E, b c d h w
            # print('shape of fea after conv: ', fea.shape)#[b c d h w]
            feat_list.append(fea)
        res = torch.cat(feat_list, dim=1)
        # print('shape of res: ', res.shape)
        return res, feat_list










