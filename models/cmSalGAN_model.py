# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_RES import resnet50
import math
import os
import cv2
import numpy as np

class UnetGAN(nn.Module):
    def __init__(self, cfg={'PicaNet': "GGLDDC",'Size': [28, 28, 28, 56, 112, 224],'Channel': [2048, 1024, 512, 256, 64, 64],
                            'loss_ratio': [0.5, 0.5, 0.5, 0.8, 0.8, 1],'block':[[64, [16]], [256, [16]], [512, [16]]]}):
        super(UnetGAN, self).__init__()
        self.encoder_rgb = resnet50()
        self.encoder_depth = resnet50()
        self.decoder_rgb = nn.ModuleList()
        self.decoder_depth = nn.ModuleList()
        self.edgeinfo = EdgeInfoLayerC(48, 64)
        block_layers = []
        for k in cfg['block']:
            block_layers += [BlockLayer(k[0], k[1])]
        self.block = nn.ModuleList(block_layers)
        self.edgeScore = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=1, padding=0))
        self.edgeMap = FuseLayer1([16, 16, 16])
        self.cfg = cfg
        for i in range(5):
            self.decoder_rgb.append(DecoderCell(size=cfg['Size'][i], in_channel=cfg['Channel'][i],
                                                out_channel=cfg['Channel'][i + 1], mode=cfg['PicaNet'][i]))
            if i == 4:
                self.decoder_rgb.append(DecoderCell(size=cfg['Size'][5], in_channel=cfg['Channel'][5],
                                                    out_channel=1, mode='C'))
        for i in range(4):
            self.decoder_depth.append(DecoderCell(size=cfg['Size'][i], in_channel=cfg['Channel'][i],
                                                  out_channel=cfg['Channel'][i + 1], mode=cfg['PicaNet'][i]))
            if i == 3:
                self.decoder_depth.append(DecoderCell(size=cfg['Size'][4], in_channel=cfg['Channel'][4],
                                                      out_channel=64, mode='D'))

    def forward(self, inlist, mode=0):
        gt = None
        rgb = inlist[0]
        en_out_rgb = self.encoder_rgb(rgb)
        dec_f = en_out_rgb[0:3]
        if mode == 0:
            gt = inlist[1]
            dec_f = [self.block[i](kk.detach()) for i, kk in enumerate(dec_f)]
            fin_e, all_elist = self.edgeMap(dec_f, gt.size())
            final_pred = np.squeeze(torch.sigmoid(fin_e[-1]).cpu().data.numpy()) * 255
            loss_list = []
            loss_fin = bce2d(fin_e, gt)
            for ix in all_elist:
                loss_list.append(bce2d(ix, gt))
            return final_pred, loss_fin + sum(loss_list)
        else:
            depth = inlist[1]
            if len(inlist) == 3:
                gt = inlist[2]
            en_out_depth = self.encoder_depth(depth)
            dec_f = [self.block[i](kk.detach()) for i, kk in enumerate(dec_f)]
            edge_merge = self.edgeinfo(dec_f, rgb.size())
            dec_rgb = None
            dec_dep = None
            pred_rlist = []
            pred_dlist = []
            for i in range(5):
                dec_rgb, _pred_rgb = self.decoder_rgb[i](en_out_rgb[4 - i], dec_rgb)
                dec_dep, _pred_dep = self.decoder_depth[i](en_out_depth[4 - i], dec_dep)
                pred_rlist.append(_pred_rgb)
                pred_dlist.append(_pred_dep)
                if i == 4:
                    dec_rgb, _pred_rgb = self.decoder_rgb[5](edge_merge, dec_rgb, dec_dep)
                    if mode == 1:
                        pred_rlist.append(_pred_rgb)
                    else:
                        finalPred = self.edgeScore(torch.cat([dec_rgb, edge_merge], dim=1))
                        finalPred = torch.sigmoid(finalPred)
                        pred_rlist.append(finalPred)
            if gt is not None:
                rgb_loss = 0
                dep_loss = 0
                for i in range(6):
                    size = pred_rlist[i].size()[2:]
                    gt_i = F.interpolate(gt, size, mode='bilinear', align_corners=True)
                    rgb_loss += F.binary_cross_entropy(pred_rlist[i], gt_i) * self.cfg['loss_ratio'][i]
                    if i < 5:
                        dep_loss += F.binary_cross_entropy(pred_dlist[i], gt_i) * self.cfg['loss_ratio'][i]
                pred_r = F.interpolate(pred_rlist[4], scale_factor=2, mode='bilinear', align_corners=True)
                pred_d = F.interpolate(pred_dlist[4], scale_factor=2, mode='bilinear', align_corners=True)
                return pred_r, pred_d, rgb_loss + dep_loss
            else:
                return pred_rlist[-1]

class DecoderCell(nn.Module):
    def __init__(self, size, in_channel, out_channel, mode):
        super(DecoderCell, self).__init__()
        self.conv1 = nn.Conv2d(2 * in_channel, in_channel, kernel_size=1, padding=0)
        self.mode = mode
        if mode == 'G':
            self.picanet = PicanetG(size, in_channel)
        elif mode == 'L':
            self.picanet = PicanetL(in_channel)
        elif mode == 'D':
            self.picanet = PicanetL(in_channel)
            self.deConv = nn.ConvTranspose2d(in_channel, in_channel, kernel_size=4, stride=2, padding=1, groups=in_channel)
        elif mode == 'C':
            self.fuseConv1 = nn.Conv2d(2 * in_channel, in_channel, kernel_size=1, padding=0)
            self.deConv_r = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, groups=64)
            self.deConv_d = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, groups=64)
            self.fuseEdgeR = nn.Conv2d(in_channel + 64, in_channel, kernel_size=3, padding=1)
            self.fuseEdgeD = nn.Conv2d(in_channel + 64, in_channel, kernel_size=3, padding=1)
            self.picanet = None
        else:
            assert 0
        if not mode == 'C':
            self.conv2 = nn.Conv2d(2 * in_channel, out_channel, kernel_size=1, padding=0)
            self.bn_feature = nn.BatchNorm2d(out_channel)
            self.conv3 = nn.Conv2d(out_channel, 1, kernel_size=1, padding=0)
        else:
            self.conv2 = nn.Conv2d(in_channel, 1, kernel_size=1, padding=0)

    def forward(self, *input):
        assert len(input) <= 3
        fmap = None
        if len(input) == 3:
            eg = input[0]
            dec1 = input[1]
            dec2 = input[2]
            dec1 = self.deConv_d(dec1)
            dec1 = F.relu(self.fuseEdgeR(torch.cat((dec1, eg), dim=1)))
            dec2 = self.deConv_r(dec2)
            dec2 = F.relu(self.fuseEdgeD(torch.cat((dec2, eg), dim=1)))
            fmap = torch.cat((dec1, dec2), dim=1)
            fmap = self.conv1(fmap)
            fmap = F.relu(fmap)
        elif input[1] is None:
            fmap = input[0]
        else:
            en = input[0]
            dec = input[1]
            if dec.size()[2] * 2 == en.size()[2]:
                dec = self.deConv(dec)
            elif dec.size()[2] != en.size()[2]:
                assert 0
            fmap = torch.cat((en, dec), dim=1)
            fmap = self.conv1(fmap)
            fmap = F.relu(fmap)
        if not self.mode == 'C':
            fmap_att = self.picanet(fmap)
            x = torch.cat((fmap, fmap_att), 1)
            x = self.conv2(x)
            x = self.bn_feature(x)
            dec_out = F.relu(x)
            _y = self.conv3(dec_out)
            _y = torch.sigmoid(_y)
        else:
            dec_out = fmap
            _y = self.conv2(fmap)
            _y = torch.sigmoid(_y)
        return dec_out, _y

class PicanetG(nn.Module):
    def __init__(self, size, in_channel):
        super(PicanetG, self).__init__()
        self.renet = Renet(size, in_channel, 100)
        self.in_channel = in_channel

    def _infer_hw_from_L(self, L, orig_h, orig_w):
        if orig_h * orig_w == L:
            return orig_h, orig_w
        if L % orig_w == 0:
            return L // orig_w, orig_w
        if L % orig_h == 0:
            return orig_h, L // orig_h
        best_pair = (1, L)
        best_score = float('inf')
        sqrtL = int(math.sqrt(L)) + 1
        for a in range(1, sqrtL):
            if L % a == 0:
                b = L // a
                score = abs((a / b) - (orig_h / orig_w))
                if score < best_score:
                    best_score = score
                    best_pair = (a, b)
        a, b = best_pair
        if (orig_h >= orig_w and a < b) or (orig_h < orig_w and a > b):
            a, b = b, a
        return a, b

    def forward(self, *input):
        x = input[0]
        b, c, h, w = x.size()
        kernel = self.renet(x)
        kernel = F.softmax(kernel, dim=1)
        Bk, Kk, Hk, Wk = kernel.size()
        kernel = kernel.view(b, Kk, -1)
        Lk = kernel.size(2)
        x_unf = F.unfold(x, kernel_size=(10, 10), dilation=(3, 3))
        K = 10 * 10
        L = x_unf.size(2)
        x_unf = x_unf.view(b, c, K, L)
        if Lk != L:
            side_k = int(round(math.sqrt(Lk)))
            if side_k * side_k != Lk:
                kernel_1d = kernel.unsqueeze(2)
                kernel_resized = F.interpolate(kernel_1d, size=(1, L), mode='bilinear', align_corners=False)
                kernel = kernel_resized.squeeze(2)
            else:
                kernel_2d = kernel.view(b, Kk, side_k, side_k)
                new_side = int(round(math.sqrt(L)))
                kernel_2d = F.interpolate(kernel_2d, size=(new_side, new_side), mode='bilinear', align_corners=False)
                kernel = kernel_2d.flatten(2, 3)
                if kernel.size(2) > L:
                    kernel = kernel[:, :, :L]
                elif kernel.size(2) < L:
                    pad_size = L - kernel.size(2)
                    pad_tensor = kernel.new_zeros(b, Kk, pad_size)
                    kernel = torch.cat([kernel, pad_tensor], dim=2)
        kernel_exp = kernel.unsqueeze(1)
        x_weighted = x_unf * kernel_exp
        x_out = torch.sum(x_weighted, dim=2)
        if L == h * w:
            H_out, W_out = h, w
        else:
            H_out, W_out = self._infer_hw_from_L(L, h, w)
        if H_out * W_out != L:
            if L % w == 0:
                H_out = L // w
                W_out = w
            elif L % h == 0:
                H_out = h
                W_out = L // h
            else:
                H_out, W_out = 1, L
        x_out = x_out.view(b, c, H_out, W_out)
        if (H_out != h) or (W_out != w):
            x_out = F.interpolate(x_out, size=(h, w), mode='bilinear', align_corners=False)
        return x_out

class PicanetL(nn.Module):
    def __init__(self, in_channel):
        super(PicanetL, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 128, kernel_size=7, dilation=2, padding=6)
        self.conv2 = nn.Conv2d(128, 49, kernel_size=1)
    def forward(self, *input):
        x = input[0]
        size = x.size()
        kernel = self.conv1(x)
        kernel = self.conv2(kernel)
        kernel = F.softmax(kernel, 1)
        kernel = kernel.reshape(size[0], 1, size[2] * size[3], 7 * 7)
        x = F.unfold(x, kernel_size=[7, 7], dilation=[2, 2], padding=6)
        x = x.reshape(size[0], size[1], size[2] * size[3], -1)
        x = torch.mul(x, kernel)
        x = torch.sum(x, dim=3)
        x = x.reshape(size[0], size[1], size[2], size[3])
        return x

class Renet(nn.Module):
    def __init__(self, size, in_channel, out_channel):
        super(Renet, self).__init__()
        self.size = size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.vertical = nn.LSTM(input_size=in_channel, hidden_size=256, batch_first=True, bidirectional=True)
        self.horizontal = nn.LSTM(input_size=512, hidden_size=256, batch_first=True, bidirectional=True)
        self.conv = nn.Conv2d(512, out_channel, 1)
    def forward(self, *input):
        x = input[0]
        temp = []
        x = torch.transpose(x, 1, 3)
        for i in range(self.size):
            h, _ = self.vertical(x[:, :, i, :])
            temp.append(h)
        x = torch.stack(temp, dim=2)
        temp = []
        for i in range(self.size):
            h, _ = self.horizontal(x[:, i, :, :])
            temp.append(h)
        x = torch.stack(temp, dim=3)
        x = torch.transpose(x, 1, 2)
        x = self.conv(x)
        return x

class EdgeInfoLayerC(nn.Module):
    def __init__(self, k_in, k_out):
        super(EdgeInfoLayerC, self).__init__()
        self.trans = nn.Sequential(nn.Conv2d(k_in, k_in, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
                                   nn.Conv2d(k_in, k_out, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
                                   nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
                                   nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False), nn.ReLU(inplace=True))
    def forward(self, x, x_size):
        tmp_x = []
        for i_x in x:
            tmp_x.append(F.interpolate(i_x, x_size[2:], mode='bilinear', align_corners=True))
        x = self.trans(torch.cat(tmp_x, dim=1))
        return x

class BlockLayer(nn.Module):
    def __init__(self, k_in, k_out_list):
        super(BlockLayer, self).__init__()
        up_in1, up_mid1, up_in2, up_mid2, up_out = [], [], [], [], []
        for k in k_out_list:
            up_in1.append(nn.Conv2d(k_in, k_in//4, 1, 1, bias=False))
            up_mid1.append(nn.Sequential(nn.Conv2d(k_in//4, k_in//4, 3, 1, 1, bias=False), nn.Conv2d(k_in//4, k_in, 1, 1, bias=False)))
            up_in2.append(nn.Conv2d(k_in, k_in//4, 1, 1, bias=False))
            up_mid2.append(nn.Sequential(nn.Conv2d(k_in//4, k_in//4, 3, 1, 1, bias=False), nn.Conv2d(k_in//4, k_in, 1, 1, bias=False)))
            up_out.append(nn.Conv2d(k_in, k, 1, 1, bias=False))
        self.block_in1 = nn.ModuleList(up_in1)
        self.block_in2 = nn.ModuleList(up_in2)
        self.block_mid1 = nn.ModuleList(up_mid1)
        self.block_mid2 = nn.ModuleList(up_mid2)
        self.block_out = nn.ModuleList(up_out)
        self.bn_1 = nn.BatchNorm2d(k_in)
        self.bn_2 = nn.BatchNorm2d(k_in)
        self.relu = nn.ReLU()
    def forward(self, x, mode=0):
        x_tmp = self.relu(self.bn_1(x + self.block_mid1[mode](self.block_in1[mode](x))))
        x_tmp = self.relu(self.bn_2(x_tmp + self.block_mid2[mode](self.block_in2[mode](x_tmp))))
        x_tmp = self.block_out[mode](x_tmp)
        return x_tmp

class FuseLayer1(nn.Module):
    def __init__(self, list_k):
        super(FuseLayer1, self).__init__()
        up = []
        for i in range(len(list_k)):
            up.append(nn.Conv2d(list_k[i], 1, 1, 1))
        self.trans = nn.ModuleList(up)
        self.fuse = nn.Conv2d(len(list_k), 1, 1, 1)
    def forward(self, list_x, x_size):
        up_x = []
        out_all = []
        for i, i_x in enumerate(list_x):
            up_x.append(F.interpolate(self.trans[i](i_x), x_size[2:], mode='bilinear', align_corners=True))
        out_fuse = self.fuse(torch.cat(up_x, dim = 1))
        for up_i in up_x:
            out_all.append(up_i)
        return out_fuse, out_all

def bce2d(input, target, reduction=None):
    assert(input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()
    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg
    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    weights = alpha * pos + beta * neg
    return F.binary_cross_entropy_with_logits(input, target, weights)
