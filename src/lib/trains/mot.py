from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from fvcore.nn import sigmoid_focal_loss_jit

from models.losses import FocalLoss, TripletLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import mot_decode
from models.utils import _sigmoid, _tranpose_and_gather_feat, _offset_and_gather_feat, _offset_maxs_and_gather_feat, _offset_near_and_gather_feat
from utils.post_process import ctdet_post_process
from .base_trainer import BaseTrainer

class MotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None # reg_loss default: l1
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        self.emb_dim = opt.reid_dim # re-ID的维度：128
        self.nID = opt.nID # nID = int(last_index + 1) 439047 怎么要这么多？
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        if opt.id_loss == 'focal':
            torch.nn.init.normal_(self.classifier.weight, std=0.01)
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.classifier.bias, bias_value)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

        # self.wid = torch.randn((opt.nID, 128)).cuda()
        self.wid = nn.Parameter(torch.randn(opt.nID, 128)).cuda()

    def forward(self, outputs, batch):
        # batch:size = 4
        # input torch.Size([4, 3, 608, 1088])
        # hm torch.Size([4, 1, 152, 272]) input.size/4
        # reg_mask torch.Size([4, 500]) max_obj = 500
        # ind torch.Size([4, 500])
        # wh torch.Size([4, 500, 4])
        # reg torch.Size([4, 500, 2])
        # ids torch.Size([4, 500])
        # bbox torch.Size([4, 500, 4])

        # outputs[i]
        # hm torch.Size([4, 1, 152, 272])
        # wh torch.Size([4, 4, 152, 272])
        # id torch.Size([4, 128, 152, 272])
        # reg torch.Size([4, 2, 152, 272])        
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks): # num_stacks: 1
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0: # 0.1
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / opt.num_stacks # ind: 代表图上的一维位置

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.id_weight > 0:
                if opt.ext_offset == 'Bi':
                    # output['id'] torch.Size([4, 128, 152, 272])
                    id_head = _offset_and_gather_feat(output['id'], batch['ind'], output['ext']) # torch.Size([4, 500, 128])
                    id_head = id_head[batch['reg_mask'] > 0].contiguous()
                    id_head = self.emb_scale * F.normalize(id_head)
                    id_target = batch['ids'][batch['reg_mask'] > 0] # 目标的真实id值,shape = [目标数]
                    id_output = self.classifier(id_head).contiguous()
                    # id_output.shape: torch.Size([该张图片上目标数， 总目标数]) [42, 439047]
                elif opt.ext_offset == 'max':
                    id_head = _offset_maxs_and_gather_feat(output['id'], batch['ind'], output['ext'], batch['hm']) # torch.Size([4, 500, 128])
                    id_head = id_head[batch['reg_mask'] > 0].contiguous()
                    id_head = self.emb_scale * F.normalize(id_head)
                    id_target = batch['ids'][batch['reg_mask'] > 0] # 目标的真实id值,shape = [目标数]
                    id_output = self.classifier(id_head).contiguous()
                elif opt.ext_offset == 'near':
                    id_head = _offset_near_and_gather_feat(output['id'], batch['ind'], output['ext']) # torch.Size([4, 500, 128])
                    id_head = id_head[batch['reg_mask'] > 0].contiguous()
                    id_head = self.emb_scale * F.normalize(id_head)
                    id_target = batch['ids'][batch['reg_mask'] > 0] # 目标的真实id值,shape = [目标数]
                    id_output = self.classifier(id_head).contiguous()
                else:
                    id_head = _tranpose_and_gather_feat(output['id'], batch['ind']) # torch.Size([4, 500, 128])
                    id_head = id_head[batch['reg_mask'] > 0].contiguous() # torch.Size([target, 128])
                    id_head = self.emb_scale * F.normalize(id_head)
                    id_target = batch['ids'][batch['reg_mask'] > 0] # 目标的真实id值,shape = [目标数]
                    id_output = self.classifier(id_head).contiguous()
                    # id_output.shape: torch.Size([该张图片上目标数， 总目标数]) [42, 439047]

                if self.opt.id_loss == 'focal':
                    id_target_one_hot = id_output.new_zeros((id_head.size(0), self.nID)).scatter_(1,
                                                                                                  id_target.long().view(
                                                                                                      -1, 1), 1)
                    id_loss += sigmoid_focal_loss_jit(id_output, id_target_one_hot,
                                                      alpha=0.25, gamma=2.0, reduction="sum"
                                                      ) / id_output.size(0)
                elif self.opt.id_loss == 'cross':
                    id_loss = 0
                    id_target_one_hot = id_output.new_zeros((id_head.size(0), self.nID)).scatter_(1,
                                                                                                  id_target.long().view(
                                                                                                      -1, 1), 1)
                    
                    id_mult = torch.mm(id_head, self.wid.t()) # [target, nID]
                    id_loss += sigmoid_focal_loss_jit(id_mult, id_target_one_hot,
                                                      alpha=0.25, gamma=2.0, reduction="mean"
                                                      ) / id_output.size(0)

                    # print(id_target_one_hot.shape)                                                                              
                    # for i in range(id_target.shape[0]):
                    #     id = id_target[i]
                    #     id_mult = (id_head[i] * self.wid).sum(axis=1) # [nID]
                    #     print(id_mult[:10])
                    #     id_sigm = 1 - torch.sigmoid(id_mult)
                    #     print(id_sigm[:10], id_sigm[id])
                    #     id_sigm[id] = 1 - id_sigm[id]
                    #     print(id_sigm[id])
                    #     id_loss += -id_sigm.log().sum()
                    #     print(id_loss)
                        
                else:
                    id_loss += self.IDLoss(id_output, id_target)

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss
        if opt.multi_loss == 'uncertainty':
            loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
            loss *= 0.5
        else:
            loss = det_loss + 0.1 * id_loss

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss}
        return loss, loss_stats


class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MotTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss']
        loss = MotLoss(opt)
        return loss_states, loss

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
