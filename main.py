import torch
from PIL import Image
import torchvision.transforms.functional as TF
import argparse
import numpy as np
from utils import fname_presuffix
from models.train import train
from models.net import occo
import os

argparser = argparse.ArgumentParser()
argparser.add_argument('--epoch', type=int, help='epoch number', default=20)
argparser.add_argument('--lr', type=float, help='task-level inner update learning rate', default=1e-4)
argparser.add_argument('--bs', type=int, help='batch size', default=10)
argparser.add_argument('--logdir', type=str, default='./logs/')
argparser.add_argument('--dataset', type=str, default='./datasets/ir/')
argparser.add_argument('--train', default=False, action='store_true')
argparser.add_argument('--test', default=False, action='store_true')
argparser.add_argument('--test_vis', type=str, help='Directory of the test visible images')
argparser.add_argument('--test_ir', type=str, help='Directory of the test infrared images')
argparser.add_argument('--ckpt', type=str, default='./logs/occo.pth')
argparser.add_argument('--use_gpu', action='store_true')
argparser.add_argument('--save_dir', type=str, default='./results/')
argparser.add_argument('--save_loss', type=str, default='./results/loss')

args = argparser.parse_args()

def RGB2YCrCb(rgb_image):
    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0, 1.0)
    Cr = Cr.clamp(0.0, 1.0).detach()
    Cb = Cb.clamp(0.0, 1.0).detach()
    return Y, Cb, Cr

def YCbCr2RGB(Y, Cb, Cr):
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
                           ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0, 1.0)
    return out

def test(args, model, vis_path, ir_path, save_path, prefix='', suffix='', flag = False):
    checkpath = args.ckpt
    print('Loading from {}...'.format(checkpath))
    vis_list = [n for n in os.listdir(vis_path)]
    ir_list = vis_list
    if args.use_gpu:
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    logs = torch.load(checkpath, map_location=device)
    model.load_state_dict(logs['state_dict'])
    model.to(device)
    import time
    Time = []
    for vis_, ir_ in zip(vis_list, ir_list):
        fn_ir = os.path.join(ir_path, ir_)
        fn_vis = os.path.join(vis_path, vis_)
        start = time.time()
        i1 = Image.open(fn_vis).convert('RGB')
        i2 = Image.open(fn_ir).convert('L')

        data_vis = TF.to_tensor(i1).unsqueeze(0).to(device)
        data_ir = TF.to_tensor(i2).unsqueeze(0).to(device)
        y, cb, cr = RGB2YCrCb(data_vis)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        output = model(y, data_ir)
        output = (output - torch.min(output)) / (torch.max(output) - torch.min(output))
        output = YCbCr2RGB(output, cb, cr)
        # print(output.shape,torch.squeeze(output, 0).shape)
        output = np.transpose((torch.squeeze(output, 0).cpu().detach().numpy() * 255.),
                              axes=(1, 2, 0)).astype(np.float32)
        save_fn = fname_presuffix(
            fname=vis_, prefix=prefix,
            suffix=suffix, newpath=save_path)
        print(save_fn)
        im = Image.fromarray(np.uint8(output))
        im.convert('RGB').save(save_fn, format='png')
        end = time.time()
        Time.append(end - start)

    print("Time: mean:%s, std: %s" % (np.mean(Time), np.std(Time)))


def main():
    print('Cuda ', torch.cuda.is_available())
    print('Training', args.train)

    # torch.cuda.set_device(1)
    if args.use_gpu:
        model = occo(nb_filter=[32, 64, 112, 160]).cuda()
    else:
        model = occo(nb_filter=[32, 64, 112, 160])

    tmp = filter(lambda x: x.requires_grad, model.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))

    print('Total trainable tensors:', num)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.train:
        model.train()
        train(model, data_path, optim, args)

    elif args.test:
        dir_vis = args.test_vis
        dir_ir = args.test_ir
        save_path = args.save_dir
        test(args, model, dir_vis, dir_ir, save_path, flag=False)

if __name__ == '__main__':
    main()