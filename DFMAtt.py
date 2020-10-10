import torch
import torch.nn as nn
import torch.nn.functional as F
#M=1 K=*
class DFMAtt(nn.Module):
    def __init__(self, in_ch,out_ch, k):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0,bias=True)
        self.k=k
        self.out_ch=out_ch
        offset_list=[]
        for x in range(k):
            conv = nn.Conv2d(in_ch, 2, 1, 1, 0,bias=True)
            offset_list.append(conv)
        self.offset_conv=nn.ModuleList(offset_list)
        self.weight_conv= nn.Sequential(nn.Conv2d(in_ch, k, 1, 1, 0,bias=True),nn.Softmax(1))

    def forward(self,input):
        b, c, h, w = input.size()
        proj_feat=self.conv(input)
        offsets=[]
        for x in range(self.k):
            flow = self.offset_conv[x](input)
            offsets.append(flow)
        offsetweights= torch.repeat_interleave(self.weight_conv(input),self.out_ch,1)
        feats=[]
        for x in range(self.k):
            flow=offsets[x]
            flow = flow.permute(0, 2, 3, 1)
            grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
            grid = torch.stack((grid_x, grid_y), 2).float()
            grid.requires_grad = False
            grid = grid.type_as(proj_feat)
            vgrid = grid + flow
            vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
            vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
            vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
            feat = F.grid_sample(proj_feat, vgrid_scaled, mode='bilinear', padding_mode='zeros')
            feats.append(feat)
        feat=torch.cat(feats,1)*offsetweights
        feat= sum(torch.split(feat,self.out_ch,1))
        return feat
