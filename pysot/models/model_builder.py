# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss_car import make_transtd_loss_evaluator
from pysot.models.backbone import get_backbone
from pysot.models.head.car_head import CARHead
from pysot.models.neck import get_neck
from ..utils.location_grid import compute_locations
from pysot.utils.xcorr import xcorr_depthwise
from pysot.models.trtd.trtd import TRTD

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        self.grader = TRTD(cfg)

        # build car head
        self.car_head = CARHead(cfg, 256)

        # build loss
        self.loss_evaluator = make_transtd_loss_evaluator(cfg)


    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)  # 模板图像为(B,3,127,127)，经过backbone和neck生成3组(B,256,7,7)的特征
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)  # 搜索图像为(B,3,255,255)，经过backbone和neck生成3组(B,256,31,31)的特征

        features = self.grader(xf, self.zf)

        # 上述features再经过car_head模块分成cls,cen和loc分支
        cls, loc, cen = self.car_head(features)
        return {
                'cls': cls,  # cls为(1,2,25,25)
                'loc': loc,  # loc为(1,4,25,25)
                'cen': cen   # cen为(1,1,25,25)
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()  # template-->(B,3,127,127)
        search = data['search'].cuda()  # search-->(B,3,255,255)
        label_cls = data['label_cls'].cuda()  # label_cls-->(B,25,25)
        label_loc = data['bbox'].cuda()  # label_loc-->(B,4)

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)  # zf长度为3，(B,256,7,7)
            xf = self.neck(xf)  # xf长度为3，(B,256,31,31)


        features = self.grader(xf, zf)

        cls, loc, cen = self.car_head(features)
        locations = compute_locations(cls, cfg.TRACK.STRIDE)
        cls = self.log_softmax(cls)
        cls_loss, loc_loss, cen_loss = self.loss_evaluator(
            locations,
            cls,
            loc,
            cen, label_cls, label_loc
        )

        # get loss
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss
        return outputs
