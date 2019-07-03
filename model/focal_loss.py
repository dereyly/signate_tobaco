import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
print(torch.__version__)
from torch.nn.modules.loss import _WeightedLoss #, #_assert_no_grad
from torch.autograd import Variable
import numpy as np
# from core.config import cfg
import pickle as pkl


def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes,device=labels.device)  # [D,D]
    return y[labels]            # [N,D]

class CrossEntropyLoss(_WeightedLoss):

    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True):
        super(CrossEntropyLoss, self).__init__(weight, size_average, reduce)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        #_assert_no_grad(target)
        return F.cross_entropy(input, target, self.weight, self.size_average,
                               self.ignore_index, self.reduce)

class DistributionLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True):
        super(DistributionLoss, self).__init__(weight, size_average)
        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, input, target):
        #_assert_no_grad(target)
        probs = nn.functional.log_softmax(input, dim=1)
        return -(target * probs).sum()

# class ReducedFocalLoss_tmp(_WeightedLoss):
#     def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True,num_cls=200):
#         super(ReducedFocalLoss_tmp, self).__init__(weight, size_average)
#         self.ignore_index = ignore_index
#         self.reduce = reduce
#         self.num_cls=num_cls
#
#     def forward(self, input, target):
#         #_assert_no_grad(target)
#         probs = F.softmax(input, dim=1)
#         pt = probs.gather(1, target.view(-1, 1)) #.requires_grad(False)
#         weight=(1-pt.data).repeat(1,self.num_cls)
#         return F.cross_entropy(input, target, weight, self.size_average,
#                         self.ignore_index, self.reduce)

import matplotlib.pyplot as plt
is_dbg=False
class ReducedFocalLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True,num_cls=200, gamma=2):
        super(ReducedFocalLoss, self).__init__(weight, size_average)
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.num_cls=num_cls
        self.gamma=gamma
        self.flat_val=0.5
        self.flat_val_pow=(1-self.flat_val)**gamma

    def forward(self, input, target, iou_weight=None):
        #_assert_no_grad(target)
        probs = F.softmax(input, dim=1)
        log_probs = F.log_softmax(input, dim=1)
        # probs_log=torch.log(probs)
        pt = probs.gather(1, target.view(-1, 1)) #.requires_grad(False)
        ##_assert_no_grad(pt)
        target_vec=one_hot_embedding(target, self.num_cls)
        batch_size=target.shape[0]
        # kpt = (1 - pt.data)
        kpt=(1 - pt.data)**self.gamma
        # a=pt.data.cpu()
        if 1:
            kpt[pt.data>self.flat_val]=self.flat_val_pow
            kpt/=self.flat_val_pow
        # plt.plot(pt.data.cpu().numpy(), kpt.cpu().numpy(), 'ro')
        # plt.show()
        # plt.savefig('/home/dereyly/ImageDB/xView/dbg/foo.png')
        if np.random.rand()<0.05 and is_dbg:
            print('focal weights',kpt[target.view(-1, 1)!=0].sum().cpu().numpy(),kpt[target.view(-1, 1)==0].sum().cpu().numpy(),kpt[target.view(-1, 1)!=0].sum().cpu().numpy()/kpt[target.view(-1, 1)==0].sum().cpu().numpy())

        if not iou_weight is None: #  and np.random.rand()>0.3 and 0:
            device_id = input.get_device()
            iou_weight[iou_weight>0.75]=0.75
            iou_w=torch.from_numpy(iou_weight**2).cuda(device_id)
            #kpt[iou_w > 0,0] *= 2*iou_w[iou_w > 0]
            kpt[iou_w > 0.65, 0] *= 4
            # if np.random.rand()>0.05:
            #     print('iou weight RPN=',iou_w[iou_w > 0].cpu().numpy().mean())
            #idx_pos=target.cpu().numpy()!=0
            zz=0

        weight = kpt.repeat(1, self.num_cls)
        if np.random.rand() > 0.2 or 1:
            weight_norm=kpt.mean().sqrt()
        else:
            weight_norm = kpt.mean()
        return -(weight*target_vec * log_probs).sum()/batch_size/weight_norm


class ReducedFocalLossSigm(_WeightedLoss):
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True,num_cls=200, gamma=2):
        super(ReducedFocalLossSigm, self).__init__()
        self.num_cls = num_cls
        self.gamma = gamma
        self.flat_val = 0.6
        self.flat_val_pow = (1 - self.flat_val) ** gamma
        self.kpt_stats=[[],[]]

    def forward(self, input, target):
        # input=input_in.cpu()
        # target=target_in.cpu()
        idx=target>=0
        if idx.sum()<2:
            return F.binary_cross_entropy_with_logits(input, target.float(), idx.float(), size_average=False)
        # a=idx.shape
        # b=input.shape
        # c=target.shape
        t = target[idx]
        is_iou=False
        if (t[t>0]<1).sum()>0:
            is_iou=True
        pt = input[idx].data.sigmoid()
        pt[t==0]=1-pt[t==0]
        # kpt = (1 - pt.data)
        kpt = (1 - pt) ** self.gamma
        if 1:
            kpt[kpt > self.flat_val_pow] = self.flat_val_pow
            kpt /= self.flat_val_pow
            if is_iou: #  and np.random.rand()>0.3 and not cfg.XVIEW.CLS_BALANCE:
                t[t>0.75]=0.75
                # kpt[t>0]*=4*t[t>0]**2
                kpt[t > 0.55] *= 2
                kpt[t > 0.7] *= 2
            # plt.plot(pt.data.cpu().numpy(), kpt.cpu().numpy(), 'ro')
            # plt.show()
            # plt.savefig('/home/dereyly/ImageDB/xView/dbg/foo.png')

            # kpt_np=kpt.cpu().numpy()
            # pt_np=pt.data.cpu().numpy()
            # for k in range(len(pt)):
            #     self.kpt_stats[0].append(pt_np[k].copy())
            #     self.kpt_stats[1].append(kpt_np[k].copy())
            # with open('/home/dereyly/progs/Detectron.pytorch/destrib_iou.pkl','wb') as f:
            #     pkl.dump(self.kpt_stats,f)
            #print('focal RPN  ',kpt[target.view(-1, 1)!=0].sum().cpu().numpy(),kpt[target.view(-1, 1)==0].sum().cpu().numpy(),kpt[target.view(-1, 1)!=0].sum().cpu().numpy()/kpt[target.view(-1, 1)==0].sum().cpu().numpy())
        weight = idx.float()
        weight[idx]=kpt
        if np.random.rand()>0.2 or 1:
            weight_norm = kpt.mean().sqrt() #*kpt.shape[0]
        else:
            weight_norm = kpt.mean()


        if 1:
            weight_norm[weight_norm < 0.1] = 0.1
            weight_norm[weight_norm > 10] = 10
            # weight /= idx.sum()*weight_norm
            # target_out=target.float()
            # print(input.dtype, target_out.dtype, weight.dtype)
            #print(kpt.shape,weight_norm)
            if np.random.rand() < 0.02 and is_dbg:
                print('focal weights RPN',kpt[t >0].sum().cpu().numpy(), kpt[t == 0].sum().cpu().numpy(),
                       kpt[t != 0].sum().cpu().numpy() / kpt[t == 0].sum().cpu().numpy())
                if is_iou and 0:
                    dbg_iou=t[t>0].cpu().numpy()
                    # idx_iou=dbg_iou < 0.4
                    # dbg_iou=dbg_iou[idx_iou]
                    # dbg_tbl=np.zeros((dbg_iou.shape[0],2))
                    # dbg_tbl[:, 1]=dbg_iou
                    # dbg_tbl[:, 0] = dbg_iou
                    print('iou loss',(t[t>0]**2).cpu().numpy().mean(),dbg_iou[dbg_iou<0.4]) #,kpt[t==0].sum(),kpt[t>0].sum())
            # weight=weight.cuda()
            # weight_norm=weight_norm.cuda()
            #if is_iou:

        target[target>0]=1
        return F.binary_cross_entropy_with_logits(input, target.float(), weight, size_average=False)/weight_norm


class FocalLoss(nn.Module):
    def __init__(self, num_classes=20):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes

    def focal_loss(self, x, y):
        '''Focal loss.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2

        t = one_hot_embedding(y.data.cpu(), 1+self.num_classes)  # [N,21]
        t = t[:,1:]  # exclude background
        t = Variable(t).cuda()  # [N,20]

        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)

    def focal_loss_alt(self, x, y):
        '''Focal loss alternative.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25

        t = one_hot_embedding(y.data.cpu(), 1+self.num_classes)
        t = t[:,1:]
        t = Variable(t).cuda()

        xt = x*(2*t-1)  # xt = x if t > 0 else -x
        pt = (2*xt+1).sigmoid()

        w = alpha*t + (1-alpha)*(1-t)
        loss = -w*pt.log() / 2
        return loss.sum()

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1,4)      # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1,4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1,self.num_classes)
        cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg])

        print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.data[0]/num_pos, cls_loss.data[0]/num_pos), end=' | ')
        loss = (loc_loss+cls_loss)/num_pos
        return loss