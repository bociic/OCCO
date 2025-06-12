import torch
from torch import nn
class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 4 // reduction, self.dim * 2),
            nn.Sigmoid())
    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max), dim=1)  # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2 B C 1 1
        return channel_weights

class SpatialWeight(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeight, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 1, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x1):
        B, _, H, W = x1.shape
        spatial_weights = self.mlp(x1).reshape(B, 1, H, W)
        return spatial_weights

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)
        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        return x1, x2

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(SelfAttention, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        x1 = (q1 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        x2 = (q2 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()

        return x1, x2


class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super(CrossPath, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.self_attn= SelfAttention(dim // reduction, num_heads=num_heads)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2 ,y1, y2 ):
        x1 = self.act(x1)
        x2 = self.act(x2)
        x1, x2 = self.self_attn(x1, x2)
        y1 = self.act(y1)
        y2 = self.act(y2)
        y1, y2 = self.cross_attn(y1, y2)
        out_x1 = self.norm1(y1 + x1)
        out_x2 = self.norm2(y2 + x2)
        return out_x1, out_x2


class CrossPath_V2(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super(CrossPath_V2, self).__init__()
        self.channel_proj3 = nn.Linear(dim, dim)
        self.channel_proj4 = nn.Linear(dim, dim)
        self.act = nn.ReLU(inplace=True)
        self.end_proj1 = nn.Linear(dim*2, dim)
        self.end_proj2 = nn.Linear(dim*2, dim)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2, c1, c2 ,s1, s2 ):
        c1 = self.act(self.channel_proj3(c1))
        c2 = self.act(self.channel_proj4(c2))
        c1, c2 = self.cross_attn(c1, c2)
        y1 = torch.cat((s1, c1), dim=-1)
        y2 = torch.cat((s2, c2), dim=-1)

        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2

class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels)
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out

class Attention_fusion(nn.Module):
        def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d):
            super(Attention_fusion, self).__init__()
            self.cross = CrossPath_V2(dim=dim, reduction=reduction, num_heads=num_heads)
            self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction,
                                            norm_layer=norm_layer)
            self.channel_weight = ChannelWeights(dim=dim, reduction=reduction)
            self.space_weight1 = SpatialWeight(dim=dim, reduction=reduction)
            self.space_weight2 = SpatialWeight(dim=dim, reduction=reduction)

        def forward(self, x1, x2):
            B, C, H, W = x1.shape
            channelweight = self.channel_weight(x1, x2)
            spaceweight1 = self.space_weight1(x1)
            spaceweight2 = self.space_weight1(x2)
            c1 = channelweight[0] * x1
            c2 = channelweight[1] * x2
            s1 = spaceweight1 * x1
            s2 = spaceweight2 * x2

            c1 = c1.flatten(2).transpose(1, 2)
            c2 = c2.flatten(2).transpose(1, 2)
            x1 = x1.flatten(2).transpose(1, 2)
            x2 = x2.flatten(2).transpose(1, 2)
            s1 = s1.flatten(2).transpose(1, 2)
            s2 = s2.flatten(2).transpose(1, 2)
            x1, x2 = self.cross(x1, x2, c1, c2, s1, s2)
            merge = torch.cat((x1, x2), dim=-1)
            merge = self.channel_emb(merge, H, W)
            return merge


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.relu   = nn.ReLU(inplace = True)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):

        inputs2 = self.up(inputs2)
        shape_x1 = inputs1.size()
        shape_x2 = inputs2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right % 2 == 0.0:
                left = int(lef_right / 2)
                right = int(lef_right / 2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot % 2 == 0.0:
                top = int(top_bot / 2)
                bot = int(top_bot / 2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        inputs2 = reflection_pad(inputs2)

        outputs = torch.cat([inputs1, inputs2], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)

        return outputs

class DownsampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.downsample = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out = self.Conv_BN_ReLU_2(x)
        out_2 = self.downsample(out)
        return out, out_2

class occo(nn.Module):
    def __init__(self, nb_filter):
        super(occo, self).__init__()

        self.v1 = DownsampleLayer(1, nb_filter[0])
        self.v2 = DownsampleLayer(nb_filter[0], nb_filter[1])  # 64-128
        self.v3 = DownsampleLayer(nb_filter[1], nb_filter[2])  # 128-256
        self.v4 = DownsampleLayer(nb_filter[2], nb_filter[3])

        self.i1 = DownsampleLayer(1, nb_filter[0])
        self.i2 = DownsampleLayer(nb_filter[0], nb_filter[1])  # 64-128
        self.i3 = DownsampleLayer(nb_filter[1], nb_filter[2])  # 128-256
        self.i4 = DownsampleLayer(nb_filter[2], nb_filter[3])

        self.fusion1 = Attention_fusion(dim=nb_filter[0], reduction=1, num_heads=4, norm_layer=nn.BatchNorm2d)
        self.fusion2 = Attention_fusion(dim=nb_filter[1], reduction=1, num_heads=4, norm_layer=nn.BatchNorm2d)
        self.fusion3 = Attention_fusion(dim=nb_filter[2], reduction=1, num_heads=4, norm_layer=nn.BatchNorm2d)
        self.fusion4 = Attention_fusion(dim=nb_filter[3], reduction=1, num_heads=1, norm_layer=nn.BatchNorm2d)

        self.up_concat3 = unetUp(nb_filter[3]+nb_filter[2], nb_filter[2])

        self.up_concat2 = unetUp(nb_filter[2]+nb_filter[1], nb_filter[1])

        self.up_concat1 = unetUp(nb_filter[1]+nb_filter[0], nb_filter[0])
        self.final = nn.Conv2d(nb_filter[0], 1, 1)

    def forward(self, x, y):
        x1, x1_0 = self.v1(x)
        x2, x2_0 = self.v2(x1_0)
        x3, x3_0 = self.v3(x2_0)
        x4, _ = self.v4(x3_0)

        y1, y1_0 = self.i1(y)
        y2, y2_0 = self.i2(y1_0)
        y3, y3_0 = self.i3(y2_0)
        y4, _ = self.i4(y3_0)

        fu1 = self.fusion1(x1, y1)
        fu2 = self.fusion2(x2, y2)
        fu3 = self.fusion3(x3, y3)
        fu4 = self.fusion4(x4, y4)

        up3 = self.up_concat3(fu3, fu4)
        up2 = self.up_concat2(fu2, up3)
        up1 = self.up_concat1(fu1, up2)

        final = self.final(up1)
        return final
