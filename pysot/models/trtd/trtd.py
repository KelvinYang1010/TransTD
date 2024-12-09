import torch.nn as nn
import torch.nn.functional as F
import torch as t
import math
from pysot.models.trtd.tran import Transformer


class TRTD(nn.Module):

    def __init__(self, cfg):
        super(TRTD, self).__init__()

        channel = 256

        self.row_embed = nn.Embedding(50, channel // 2)
        self.col_embed = nn.Embedding(50, channel // 2)
        self.reset_parameters()

        self.transformer = Transformer(channel, 8, 1, 2)  # nhead=8, num_encoder_layers=1, num_decoder_layers=2

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def xcorr_depthwise(self, x, kernel):
        """depthwise cross correlation
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch * channel, x.size(2), x.size(3))  # x的shape为(1,384,30,30)
        kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch * channel)
        out = out.view(batch, channel, out.size(2), out.size(3))  # out的shape为(1,384,21,21)
        return out

    def forward(self, x, z):

        # res1->(B,256,25,25),x[0]->(B,256,31,31),z[0]->(B,256,7,7)
        res1 = self.xcorr_depthwise(x[0], z[0])
        # res2->(B,256,25,25),x[1]->(B,256,31,31),z[1]->(B,256,7,7)
        res2 = self.xcorr_depthwise(x[1], z[1])
        # res3->(B,256,25,25),x[2]->(B,256,31,31),z[2]->(B,256,7,7)
        res3 = self.xcorr_depthwise(x[2], z[2])

        h, w = res3.shape[-2:]
        i = t.arange(w).cuda()
        j = t.arange(h).cuda()
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)

        # pos为位置编码，shape为(B,256,25,25)
        pos = t.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(res3.shape[0], 1, 1, 1)

        # b=batch_size，c=256，w=25，h=25
        b, c, w, h = res3.size()
        # 经过transformer后得到(25*25,B,256)的特征
        res = self.transformer((pos + res1).view(b, c, -1).permute(2, 0, 1), \
                               (pos + res2).view(b, c, -1).permute(2, 0, 1), \
                               res3.view(b, c, -1).permute(2, 0, 1))

        res = res.permute(1, 2, 0).view(b, c, w, h)  # res的形状为(B,256,25,25)

        return res





