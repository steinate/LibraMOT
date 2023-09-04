import torch
import torch.nn.functional as F

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

def _offset_and_gather_feat(feat, ind, offset):
    # feat torch.Size([4, 128, 152, 272])
    # ind torch.Size([4, 500])
    # offset torch.Size([4, 2, 152, 272])
    batch_size = feat.shape[0]
    h, w = feat.shape[2], feat.shape[3]
    num_obj = torch.count_nonzero(ind, dim=1)
    feat = feat.permute(0, 2, 3, 1).contiguous() # torch.Size([4, 152, 272, 128])
    offset = offset.permute(0, 2, 3, 1).contiguous() # torch.Size([4, 152, 272, 2])
    dim  = offset.size(3) # 2
    
    ind_ = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim) # ind_ ([4, 500, 2]) 最后2个维度：重新复制一次
    offset = offset.view(offset.size(0), -1, offset.size(3)) # torch.Size([4, 152*272, 2])
    offset = offset.gather(1, ind_) # torch.Size([4, 500, 2])
    eps = 0.0001
    row = ind // w + offset[:,:,0]
    row = torch.clamp(row, 0, h-1-eps).unsqueeze(1)
    col = ind % w + offset[:,:,1]
    col = torch.clamp(col, 0, w-1-eps).unsqueeze(1) # torch.Size([4, 1, 500])

    row = (row - h/2) / (h/2)
    col = (col - w/2) / (w/2)

    b_sample = torch.stack([row, col], dim=3) # torch.Size([4, 1, 500, 2])
    feat_ = F.grid_sample(feat, b_sample, align_corners=True)

    # feat_ = torch.zeros((batch_size, 500, 128))

    # for b in range(num_obj.shape[0]):
    #   for i in range(num_obj[b]):
    #     row_off = row[b, i] % 1
    #     row_ind = int(row[b, i] // 1)
    #     # row_ind = row[b, i].floor()
    #     col_off = col[b, i] % 1
    #     col_ind = int(col[b, i] // 1)
    #     # col_ind = col[b, i].floor()
    #     feat_[b, i, :] = feat[b, row_ind, col_ind, :]*(1-row_off)*(1-col_off) + \
    #                     feat[b, row_ind + 1, col_ind, :]*row_off*(1-col_off) + \
    #                     feat[b, row_ind, col_ind + 1, :]*(1-row_off)*col_off + \
    #                     feat[b, row_ind + 1, col_ind + 1, :]*row_off*col_off
    return feat_

def matrix_offset(matrix, offset_x, offset_y):
        b, c, h, w = matrix.shape
        offset_matrix = torch.zeros((b, c, h+2, w+2), device=matrix.device)
        offset_matrix[:, :, 1+offset_x:h+offset_x+1, 1+offset_y:w+offset_y+1] = matrix
        return offset_matrix[:, :, 1:h+1, 1:w+1]
    
def offset_matrix(matrix):
    cat_matrix = matrix
    for i in range(-1,2):
        for j in range(-1,2):
            if  (not i and not j):
                continue
            cat_matrix = torch.cat((cat_matrix, matrix_offset(matrix, i, j)), 1)
    return cat_matrix

path = '/media/zyf301/AE1EF53A1EF4FBE1/pengjq/FairMOT/exp/mot/finetune_finetune_finetune/model_last.pth'
model_dict=torch.load(path)
for param_tensor in model_dict['state_dict']:
    #打印 key value字典
    # print(param_tensor,'\t',model_dict.state_dict()[param_tensor].size())
    print(param_tensor)


# python track.py mot --alltest_mot17 True --load_model ../exp/mot/finetune_extloss2siaminput/model_20.pth --conf_thres 0.4 --gpus 1 --siam_input 'double_hm' --id_loss 'cross' --ext_offset 'Bi'
# python track.py mot --val_mot17 True --load_model ../exp/mot/baseline_mot17/model_last.pth --conf_thres 0.4 --gpus 1