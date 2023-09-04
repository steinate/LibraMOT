from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _gather_feat(feat, ind, mask=None):
    # hm torch.Size([4, 152*272, 1])
    dim  = feat.size(2) # 1(类别数)
    # ind torch.Size([4, 500])
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    # unsqueeze([4, 500, 128]) 
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat # torch.Size([4, 500, 128])

def _tranpose_and_gather_feat(feat, ind):
    # ind torch.Size([4, 500])
    # hm torch.Size([4, 1, 152, 272])
    feat = feat.permute(0, 2, 3, 1).contiguous()
    # hm torch.Size([4, 152, 272, 1])
    feat = feat.view(feat.size(0), -1, feat.size(3))
    # hm torch.Size([4, 152*272, 1])
    feat = _gather_feat(feat, ind)
    return feat # torch.Size([4, 500, 128])

def _offset_and_gather_feat(feat, ind, offset):
    # feat torch.Size([4, 128, 152, 272])
    # ind torch.Size([4, 500])
    # offset torch.Size([4, 2, 152, 272])
    batch_size = feat.shape[0]
    num_obj = torch.count_nonzero(ind, dim=1)
    h, w = feat.shape[2], feat.shape[3]
    feat = feat.permute(0, 2, 3, 1).contiguous()
    # torch.Size([4, 152, 272, 128])
    offset = offset.permute(0, 2, 3, 1).contiguous()
    # torch.Size([4, 152, 272, 2])
    dim  = offset.size(3) # 1(类别数)
    ind_ = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    # ind_ ([4, 500, 2]) 最后2个维度：重新复制一次
    offset = offset.view(offset.size(0), -1, offset.size(3))
    # torch.Size([4, 152*272, 2])
    offset = offset.gather(1, ind_)
    # torch.Size([4, 500, 2])
    eps = 0.0001
    row = ind // w + offset[:,:,0]
    row = torch.clamp(row, 0, h-1-eps)
    col = ind % w + offset[:,:,1]
    col = torch.clamp(col, 0, w-1-eps)
    # torch.Size([4, 500])

    feat_ = torch.zeros((batch_size, 500, 128))

    for b in range(num_obj.shape[0]):
      for i in range(num_obj[b]):
        

        row_off = row[b, i] % 1
        row_ind = int(row[b, i] // 1)
        # row_ind = row[b, i].floor()
        col_off = col[b, i] % 1
        col_ind = int(col[b, i] // 1)
        # col_ind = col[b, i].floor()
        row_add = int(row_off>=0.5)
        col_add = int(col_off>=0.5)

        # feat_[b, i, :] = feat[b, row_ind + row_add, col_ind + col_add, :]
        feat_[b, i, :] = feat[b, row_ind, col_ind, :]*(1-row_off)*(1-col_off) + \
                        feat[b, row_ind + 1, col_ind, :]*row_off*(1-col_off) + \
                        feat[b, row_ind, col_ind + 1, :]*(1-row_off)*col_off + \
                        feat[b, row_ind + 1, col_ind + 1, :]*row_off*col_off
        # if b == 0 and i == 0:
        #   print(feat[b, row_ind, col_ind, 0:10])
        #   print(feat[b, row_ind + 1, col_ind, 0:10])
        #   print(feat[b, row_ind, col_ind + 1, 0:10])
        #   print(feat[b, row_ind + 1, col_ind + 1, 0:10])
        #   print(feat_[b, i, :])
    return feat_.cuda()

def _offset_near_and_gather_feat(feat, ind, offset):
    # feat torch.Size([4, 128, 152, 272])
    # ind torch.Size([4, 500])
    # offset torch.Size([4, 2, 152, 272])
    batch_size = feat.shape[0]
    num_obj = torch.count_nonzero(ind, dim=1)
    h, w = feat.shape[2], feat.shape[3]
    feat = feat.permute(0, 2, 3, 1).contiguous()
    # torch.Size([4, 152, 272, 128])
    offset = offset.permute(0, 2, 3, 1).contiguous()
    # torch.Size([4, 152, 272, 2])
    dim  = offset.size(3) # 1(类别数)
    ind_ = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    # ind_ ([4, 500, 2]) 最后2个维度：重新复制一次
    offset = offset.view(offset.size(0), -1, offset.size(3))
    # torch.Size([4, 152*272, 2])
    offset = offset.gather(1, ind_)
    # torch.Size([4, 500, 2])
    eps = 0.0001
    row = ind // w + offset[:,:,0]
    row = torch.clamp(row, 0, h-1-eps)
    col = ind % w + offset[:,:,1]
    col = torch.clamp(col, 0, w-1-eps)
    # torch.Size([4, 500])

    feat_ = torch.zeros((batch_size, 500, 128))

    for b in range(num_obj.shape[0]):
      for i in range(num_obj[b]):
        

        row_off = row[b, i] % 1
        row_ind = int(row[b, i] // 1)
        # row_ind = row[b, i].floor()
        col_off = col[b, i] % 1
        col_ind = int(col[b, i] // 1)
        # col_ind = col[b, i].floor()
        row_add = int(row_off>=0.5)
        col_add = int(col_off>=0.5)

        feat_[b, i, :] = feat[b, row_ind + row_add, col_ind + col_add, :]
    return feat_.cuda()

def _offset_maxs_and_gather_feat(feat, ind, offset, score):
    # feat torch.Size([4, 128, 152, 272])
    # ind torch.Size([4, 500])
    # offset torch.Size([4, 2, 152, 272])
    # score torch.Size([4, 1, 152, 272])
    batch_size = feat.shape[0]
    num_obj = torch.count_nonzero(ind, dim=1)
    h, w = feat.shape[2], feat.shape[3]
    feat = feat.permute(0, 2, 3, 1).contiguous()
    # torch.Size([4, 152, 272, 128])
    offset = offset.permute(0, 2, 3, 1).contiguous()
    # torch.Size([4, 152, 272, 2])
    dim  = offset.size(3) # 1(类别数)
    ind_ = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    # ind_ ([4, 500, 2]) 最后2个维度：重新复制一次
    offset = offset.view(offset.size(0), -1, offset.size(3))
    # torch.Size([4, 152*272, 2])
    offset = offset.gather(1, ind_)
    # torch.Size([4, 500, 2])
    eps = 0.001
    row = ind // w + offset[:,:,0]
    row = torch.clamp(row, 0, h-1-eps)
    col = ind % w + offset[:,:,1]
    col = torch.clamp(col, 0, w-1-eps)
    # torch.Size([4, 500])

    feat_ = torch.zeros((batch_size, 500, 128))

    for b in range(num_obj.shape[0]):
      for i in range(num_obj[b]):
        row_ind = int(row[b, i])
        col_ind = int(col[b, i])
        round_scores = torch.Tensor([score[b, 0, row_ind, col_ind], score[b, 0, row_ind + 1, col_ind], \
                                  score[b, 0, row_ind, col_ind + 1], score[b, 0, row_ind + 1, col_ind + 1]])
        round_max = torch.argmax(round_scores)
        feat_[b, i, :] = feat[b, row_ind + round_max%2, col_ind + round_max//2, :]

        # if b == 0 and i == 0:
        #   print(round_scores)
        #   print(round_max)
        #   print(feat[b, row_ind, col_ind, 0:10])
        #   print(feat[b, row_ind + 1, col_ind, 0:10])
        #   print(feat[b, row_ind, col_ind + 1, 0:10])
        #   print(feat[b, row_ind + 1, col_ind + 1, 0:10])
        #   print(feat_[b, i, 0:10])
    return feat_.cuda()



def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2, 
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)