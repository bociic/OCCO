import torch
from torchvision import models
from torch import nn
import glob
import os
import contextual_loss as cl
from torch.nn import functional as F_torch

def load_model(model_name, model_dir):
    model = eval('models.%s(init_weights=False)' % model_name)
    path_format = os.path.join(model_dir, '%s-[a-z0-9]*.pth' % model_name)

    model_path = glob.glob(path_format)[0]
    model = model.cuda()

    model.load_state_dict(torch.load(model_path))
    for param in model.parameters():
        param.requires_grad = False
    return model


class Vgg19(torch.nn.Module):
    """ First layers of the VGG 19 model for the VGG loss.
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        model_path (str): Path to model weights file (.pth)
        requires_grad (bool): Enables or disables the "requires_grad" flag for all model parameters
    """

    def __init__(self, requires_grad: bool = False, vgg19_weights=None):
        super(Vgg19, self).__init__()
        if vgg19_weights is None:
            vgg_pretrained_features = load_model('vgg19', '../').features
        else:
            model = models.vgg19(pretrained=True)
            pretrain_dict = model.state_dict()
            layer1 = pretrain_dict['features.0.weight']

            new = torch.zeros(64, 1, 3, 3)
            for i, output_channel in enumerate(layer1):
                # Grey = 0.299R + 0.587G + 0.114B, RGB2GREY
                new[i] = 0.299 * output_channel[0] + 0.587 * output_channel[1] + 0.114 * output_channel[2]
            pretrain_dict['features.0.weight'] = new
            model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            model.load_state_dict(pretrain_dict)
            vgg_pretrained_features = model.features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])#
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])#
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class Vgg19_Unet(torch.nn.Module):

    def __init__(self, requires_grad: bool = False, vgg19_weights=None):
        super(Vgg19_Unet, self).__init__()
        if vgg19_weights is None:
            vgg_pretrained_features = load_model('vgg19', '../').features
        else:
            model = models.vgg19(pretrained=True)
            pretrain_dict = model.state_dict()
            layer1 = pretrain_dict['features.0.weight']
            # print(layer1.shape)
            new = torch.zeros(64, 1, 3, 3)
            for i, output_channel in enumerate(layer1):
                # Grey = 0.299R + 0.587G + 0.114B, RGB2GREY
                new[i] = 0.299 * output_channel[0] + 0.587 * output_channel[1] + 0.114 * output_channel[2]
            pretrain_dict['features.0.weight'] = new
            model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            model.load_state_dict(pretrain_dict)
            vgg_pretrained_features = model.features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        self.slice1.add_module(str(2), nn.MaxPool2d(2, 2))
        for x in range(2, 4):
            self.slice1.add_module(str(x+1), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x+1), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.slice3.add_module(str(x+1), vgg_pretrained_features[x])#

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)

        return h_relu1, h_relu2, h_relu3


class PerceptualLoss(torch.nn.Module):
    """ Defines a criterion that captures the high frequency differences between two images.
    `"Perceptual Losses for Real-Time Style Transfer and Super-Resolution" <https://arxiv.org/pdf/1603.08155.pdf>`_
    Args:
        model_path (str): Path to model weights file (.pth)
    """
    def __init__(self, vgg19_weights=None):
        super(PerceptualLoss, self).__init__()
        self.vgg = Vgg19(vgg19_weights=vgg19_weights)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss



class ContrastiveLoss(torch.nn.Module):
    def __init__(self, vgg19_weights=None):
        super(ContrastiveLoss, self).__init__()
        self.vgg = Vgg19(vgg19_weights=vgg19_weights)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]


    def forward(self, negative, output, positive):
        n_vgg, p_vgg, o_vgg = self.vgg(negative), self.vgg(positive), self.vgg(output)
        loss = 0
        for i in range(len(o_vgg)):
            loss += self.weights[i] * self.criterion(o_vgg[i], p_vgg[i].detach())/(self.criterion(o_vgg[i], n_vgg[i])+self.criterion(o_vgg[i], n_vgg[i]))
        # print('contrastive loss',loss)
        return loss
class ContrastiveLoss_L1(torch.nn.Module):
    """
    Contrastive loss with multiple negative samples
    """
    def __init__(self, vgg19_weights=None):
        super(ContrastiveLoss_L1, self).__init__()
        self.vgg = Vgg19(vgg19_weights=vgg19_weights)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
    def forward(self, negative_1, negative_2, negative_3, negative_4, output, positive):
        n1_vgg, n2_vgg, n3_vgg, n4_vgg, p_vgg, o_vgg = self.vgg(negative_1), self.vgg(negative_2), self.vgg(negative_3), self.vgg(negative_4), self.vgg(positive), self.vgg(output)
        loss = 0
        for i in range(len(o_vgg)):
            if i > 2:
                loss += self.weights[i] * self.criterion(o_vgg[i], p_vgg[i].detach()) /      \
                        (self.criterion(o_vgg[i], n1_vgg[i])+self.criterion(o_vgg[i], n2_vgg[i]) +
                         +self.criterion(o_vgg[i], n3_vgg[i])+self.criterion(o_vgg[i], n4_vgg[i]))
        return loss
class ContrastiveLoss_bg(torch.nn.Module):
    """
    Contrastive loss with multiple negative samples
    """
    def __init__(self, vgg19_weights=None):
        super(ContrastiveLoss_bg, self).__init__()
        self.vgg = Vgg19(vgg19_weights=vgg19_weights)
        self.criterion = nn.MSELoss()
        self.weights = [1.0/16, 1.0/16, 1.0/8, 1.0/4, 1.0]
    def forward(self, negative_1, negative_2, negative_3, negative_4, output, positive):
        n1_vgg, n2_vgg, n3_vgg, n4_vgg, p_vgg, o_vgg = self.vgg(negative_1), self.vgg(negative_2), self.vgg(negative_3), self.vgg(negative_4), self.vgg(positive), self.vgg(output)
        loss = 0
        for i in range(len(o_vgg)):
            if i<1:
                loss += self.weights[i] * self.criterion(o_vgg[i], p_vgg[i].detach()) /      \
                        (self.criterion(o_vgg[i], n1_vgg[i])+self.criterion(o_vgg[i], n2_vgg[i]) +
                         +self.criterion(o_vgg[i], n3_vgg[i])+self.criterion(o_vgg[i], n4_vgg[i]))
        return loss

class ContrastiveLoss_L2(torch.nn.Module):
    """
    Contrastive loss with multiple negative samples
    """
    def __init__(self, vgg19_weights=None):
        super(ContrastiveLoss_L2, self).__init__()
        self.vgg = Vgg19(vgg19_weights=vgg19_weights)
        self.criterion = nn.MSELoss()
        self.weights = [1.0/16, 1.0/16, 1.0/8, 1.0/4, 1.0]
    def forward(self, negative_1, negative_2, negative_3, negative_4, output, positive):
        n1_vgg, n2_vgg, n3_vgg, n4_vgg, p_vgg, o_vgg = self.vgg(negative_1), self.vgg(negative_2), self.vgg(negative_3), self.vgg(negative_4), self.vgg(positive), self.vgg(output)
        loss = 0
        for i in range(len(o_vgg)):
            if i>2:
                loss += self.weights[i] * self.criterion(o_vgg[i], p_vgg[i].detach()) /      \
                        (self.criterion(o_vgg[i], n1_vgg[i])+self.criterion(o_vgg[i], n2_vgg[i]) +
                         +self.criterion(o_vgg[i], n3_vgg[i])+self.criterion(o_vgg[i], n4_vgg[i]))
        return loss
    # def forward(self, negative_1, negative_2, negative_3, output, positive):
    #     n1_vgg, n2_vgg, n3_vgg, p_vgg, o_vgg = self.vgg(negative_1), self.vgg(negative_2), self.vgg(negative_3), self.vgg(positive), self.vgg(output)
    #     loss = 0
    #     for i in range(len(o_vgg)):
    #         if i < 1:
    #             loss += self.weights[i] * self.criterion(o_vgg[i], p_vgg[i].detach()) /      \
    #                     (self.criterion(o_vgg[i], n1_vgg[i])+self.criterion(o_vgg[i], n2_vgg[i]) +
    #                      +self.criterion(o_vgg[i], n3_vgg[i]))
    #     return loss


class ContrastiveLoss_multiNegative(torch.nn.Module):
    def __init__(self):
        super(ContrastiveLoss_multiNegative, self).__init__()
        self.criterion1 = cl.ContextualBilateralLoss(use_vgg=True, vgg_layer='relu5_4')
        self.criterion2 = cl.ContextualBilateralLoss(use_vgg=True, vgg_layer='relu4_4')

        # self.criterion1=cl.ContextualLoss(use_vgg=True, vgg_layer='relu5_4')
        # self.criterion2 = cl.ContextualLoss(use_vgg=True, vgg_layer='relu4_4')

        # self.criterion3 = cl.ContextualBilateralLoss(use_vgg=True, vgg_layer='relu3_4')
        # self.criterion4 = cl.ContextualBilateralLoss(use_vgg=True, vgg_layer='relu2_2')
        # self.criterion5 = cl.ContextualBilateralLoss(use_vgg=True, vgg_layer='relu1_2')
    def forward(self, negative_1, negative_2, negative_3, negative_4, output, positive):
        negative_1 = negative_1.repeat(1, 3, 1, 1)
        negative_2 = negative_2.repeat(1, 3, 1, 1)
        negative_3 = negative_3.repeat(1, 3, 1, 1)
        negative_4 = negative_4.repeat(1, 3, 1, 1)
        output = output.repeat(1, 3, 1, 1)
        positive = positive.repeat(1, 3, 1, 1)
        loss = self.criterion1(positive, output) / (
                    self.criterion1(output,negative_1 ) + self.criterion1(output, negative_2)
                    + self.criterion1(output,negative_3 ) + self.criterion1(output,negative_4 ))
        loss += 0.25 * self.criterion2(output,positive ) / (
                self.criterion2(output, negative_1) + self.criterion2(output, negative_2)
                + self.criterion2(output, negative_3) + self.criterion2(output, negative_4))

        return loss


class cbloss(torch.nn.Module):
    def __init__(self):
        super(cbloss, self).__init__()
    def forward(self, x, y):
        loss = cbl(x,y)
        return loss

def cbl(x, y):
    device = x.device
    # Calculate two image spatial loss
    grid = _compute_meshgrid(x.shape).to(device)
    distance = _compute_mse_distance(grid, grid)
    relative_distance = _compute_relative_distance(distance)
    contextual_spatial = _compute_contextual(relative_distance)

    # Calculate feature loss
    # Calculate two image distance
    distance = _compute_cosine_distance(x, y)
    # Compute relative distance
    relative_distance = _compute_relative_distance(distance)
    # Calculate two image contextual loss
    contextual_feature = _compute_contextual(relative_distance)

    # Combine loss
    cx_combine = (1. - .5) * contextual_feature + .5 * contextual_spatial
    k_max_NC, _ = torch.max(cx_combine, dim=2, keepdim=True)
    cx = k_max_NC.mean(dim=1)
    loss = torch.mean(-torch.log(cx + 1e-5))

    return loss

def _compute_meshgrid(shape) :
    batch_size, _, height, width = shape

    rows = torch.arange(0, height, dtype=torch.float32) / (height + 1)
    cols = torch.arange(0, width, dtype=torch.float32) / (width + 1)

    feature_grid = torch.meshgrid(rows, cols, indexing="ij")
    feature_grid = torch.stack(feature_grid).unsqueeze(0)
    feature_grid = torch.cat([feature_grid for _ in range(batch_size)], dim=0)

    return feature_grid

def _compute_cosine_distance(x, y):
    batch_size, channels, _, _ = x.size()

    # mean shifting by channel-wise mean of `y`.
    y_mean = y.mean(dim=(0, 2, 3), keepdim=True)
    x_centered = x - y_mean
    y_centered = y - y_mean

    # L2 normalization
    x_normalized = F_torch.normalize(x_centered, p=2, dim=1)
    y_normalized = F_torch.normalize(y_centered, p=2, dim=1)

    # Channel-wise vectorization
    x_normalized = x_normalized.reshape(batch_size, channels, -1)  # (N, C, H*W)
    y_normalized = y_normalized.reshape(batch_size, channels, -1)  # (N, C, H*W)

    # cosine similarity
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)  # (N, H*W, H*W)

    # convert to distance
    distance = 1 - cosine_sim

    return distance


def _compute_mae_distance(x, y):
    batch_size, channels, height, width = x.size()

    x_vec = x.view(batch_size, channels, -1)
    y_vec = y.view(batch_size, channels, -1)

    distance = x_vec.unsqueeze(2) - y_vec.unsqueeze(3)
    distance = distance.sum(dim=1).abs()
    distance = distance.transpose(1, 2).reshape(batch_size, height * width, height * width)
    distance = distance.clamp(min=0.)

    return distance


def _compute_mse_distance(x, y) :
    batch_size, channels, height, width = x.size()

    x_vec = x.view(batch_size, channels, -1)
    y_vec = y.view(batch_size, channels, -1)
    x_s = torch.sum(x_vec ** 2, dim=1)
    y_s = torch.sum(y_vec ** 2, dim=1)
    batch_size, HW = y_s.shape

    A = y_vec.transpose(1, 2) @ x_vec  # N x(HW) x (HW)
    distance = y_s.unsqueeze(dim=2) - 2 * A + x_s.unsqueeze(dim=1)
    distance = distance.transpose(1, 2).reshape(batch_size, height * width, height * width)
    distance = distance.clamp(min=0.)

    return distance


def _compute_relative_distance(distance):
    dist_min, _ = torch.min(distance, dim=2, keepdim=True)
    relative_distance = distance / (dist_min + 1e-5)

    return relative_distance


def _compute_contextual(relative_distance):
    # This code easy OOM
    # w = torch.exp((1 - relative_distance) / bandwidth)  # Eq(3)
    # contextual = w / torch.sum(w, dim=2, keepdim=True)  # Eq(4)

    # This code is safe
    contextual = F_torch.softmax((1 - relative_distance) , dim=2)

    return contextual


