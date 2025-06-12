import random
from pytorch_ssim import ssim
from utils.utils import *
from utils.checkpoint import save_min
from models.P_loss import ContrastiveLoss_multiNegative, ContrastiveLoss_bg
from utils import data_load
import torchvision.transforms as t

device = torch.device(1)
def RGB2YCrCb(rgb_image):
    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0,1.0)
    Cr = Cr.clamp(0.0,1.0).detach()
    Cb = Cb.clamp(0.0,1.0).detach()
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
    out = out.clamp(0,1.0)
    return out

def train(model, data_path, optimizer, args):
	checkpath = args.logdir
	if not os.path.exists(checkpath):
		os.makedirs(checkpath)

	print('total e:', args.epoch)
	e = 0
	if args.resume:
		logs = torch.load(args.resume_ckpt)
		model.load_state_dict(logs['state_dict'])
		optimizer.load_state_dict(logs['optimizer'])

	print("-----------------Parameters--------------------")
	print('Epoch: ', args.epoch)
	print('Checkpoint save path: ', checkpath)
	print("-----------------------------------------------")
	ite_num = 0
	min=9999.
	model = model.to(device)
	for i in range(args.epoch):
		train_ep(
			epoch_idx=i, 
			model=model, 
			path = data_path,
			ite_num=ite_num, 
			optimizer=optimizer,
			args=args,
			min=min
		)


def loaddata(path):
	imgs_paths_ir, names = data_load.list_images(path)
	random.shuffle(imgs_paths_ir)
	imgs_paths_vi = [x.replace('ir', 'vi') for x in imgs_paths_ir]
	randomcrop = t.RandomCrop(64)
	num = len(imgs_paths_ir)
	batch = []
	ii = 0
	batches = []
	for i in range(num):
		infrared_path = imgs_paths_ir[i]
		visible_path = imgs_paths_vi[i]
		img_vi = data_load.get_test_image(visible_path, flag=True)
		img_ir = data_load.get_test_image(infrared_path, flag=False)
		img = torch.concat([img_vi, img_ir], dim=1)
		for k in range(36):
			img_64 = randomcrop(img)
			if (ii == 0):
				batch = img_64
			else:
				batch = torch.concat([batch, img_64], dim=0)
			ii += 1
			if (ii == 100):
				batches.append(batch)
				ii = 0
				batch = []
	num = len(batches)
	return batches, num


def loadmask(path):
	imgs_paths_ir, _ = data_load.list_images(path)
	random.shuffle(imgs_paths_ir)

	imgs_paths_vi = [x.replace('ir', 'vi') for x in imgs_paths_ir]
	mask_paths_ir = [x.replace('ir/', 'mask_ir/mask_') for x in imgs_paths_ir]
	mask_paths_vi = [x.replace('ir/', 'mask_vi/mask_') for x in imgs_paths_ir]
	randomcrop = t.RandomCrop(256)
	num = len(imgs_paths_ir)
	batch = []
	ii = 0
	batches = []
	for i in range(num):
		infrared_path = imgs_paths_ir[i]
		visible_path = imgs_paths_vi[i]
		infrared_mask = mask_paths_ir[i]
		visible_mask = mask_paths_vi[i]

		img_vi = data_load.get_test_image(visible_path, flag=True)
		img_ir = data_load.get_test_image(infrared_path, flag=False)
		mask_vi = data_load.get_test_image(visible_mask, flag=False)
		mask_ir = data_load.get_test_image(infrared_mask, flag=False)
		if torch.max(mask_vi) == 0 and torch.max(mask_ir) == 0:
			continue
		if torch.max(mask_vi) == 0:
			mask_vi = torch.zeros_like(img_ir, dtype=torch.float)
		elif torch.max(mask_ir) == 0:
			mask_ir = torch.zeros_like(img_ir, dtype=torch.float)
		img = torch.concat([img_vi, img_ir, mask_vi, mask_ir], dim=1)
		k = 0
		n = 0
		while k < 2:
			n += 1
			if n == 20:
				k = 8
			img1 = randomcrop(img)
			if torch.max(img1[:, 4, :, :]) == 0 and torch.max(img1[:, 5, :, :]) == 0:
				continue
			else:
				k += 1
				if ii == 0:
					batch = img1
				else:
					batch = torch.concat([batch, img1], dim=0)
				ii += 1
				if ii == 12:
					batches.append(batch)
					ii = 0
					batch = []
	num = len(batches)
	return batches, num


def train_ep(epoch_idx, model, path, ite_num, optimizer,  args, min):
	data, number = loadmask(path)
	total = 0.
	for batch_idx in range(number):
		batch = data[batch_idx]
		ite_num = ite_num + 1
		x1 = batch[:, 0:3, :, :]
		ir = batch[:, 3, :, :]
		mask_1 = batch[:, 4, :, :]
		mask_2 = batch[:, 5, :, :]
		vi, cb, cr = RGB2YCrCb(x1)
		n, w, h = ir.shape[0], ir.shape[1], ir.shape[2]

		vi = vi.view([n, 1, w, h]).to(device)
		ir = ir.view([n, 1, w, h]).to(device)
		mask_1 = mask_1.view([n, 1, w, h]).to(device)
		mask_2 = mask_2.view([n, 1, w, h]).to(device)

		f = model(vi, ir)

		f = (f - torch.min(f)) / (torch.max(f) - torch.min(f))

		unique = ContrastiveLoss_multiNegative().to(device)
		vgg19_weights = 'placeholder'
		bg = ContrastiveLoss_bg(vgg19_weights).to(device)

		mask_share = mask_1 * mask_2
		m1 = mask_1 - mask_share
		m2 = mask_2 - mask_share
		mbg = 1 - m1 - m2 - mask_share

		lg = grad(vi, ir, f)
		ls = (1 - ssim(vi, f)) + (1 - ssim(ir, f))
		li = Fusion_loss(vi, ir, f)
		lossp = 1*ls + 10*li + 1*lg

		contrast_loss = contrastiveLoss(vi, ir, f, m1, unique, int(n / 4))
		contrast_loss += contrastiveLoss(ir, vi, f, m2, unique, int(n / 4))
		contrast_loss += 0.5 * contrastiveLoss(vi, ir, f, mask_share, unique, int(n / 4))
		contrast_loss += 0.5 * contrastiveLoss(ir, vi, f, mask_share, unique, int(n / 4))
		contrast_loss += contrastiveLoss(vi, ir, f, mbg, bg, int(n / 4))
		contrast_loss = 10*contrast_loss
		loss = lossp + contrast_loss

		total += loss
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if float(loss) < min:
			save_min(model, optimizer, args.logdir, epoch_idx, ite_num)
		if batch_idx % 10 == 0:
			print('epoch: {}, batch: {}, contrast_loss: {},total_loss: {}'.format(epoch_idx, batch_idx, contrast_loss, loss))
	print('mean loss:', total / number)


def contrastiveLoss(x1, x2, y, m, c_loss, n):
	m1 = m[0*n:1*n, :, :, :]
	m2 = m[1*n:2*n, :, :, :]
	m3 = m[2*n:3*n, :, :, :]
	m4 = m[3*n:4*n, :, :, :]

	y = y[0*n:1*n, :, :, :] * m1
	pos = x1[0*n:1*n, :, :, :] * m1
	neg1 = x2[0*n:1*n, :, :, :] * m1
	neg2 = x2[1*n:2*n, :, :, :] * m2
	neg3 = x2[2*n:3*n, :, :, :] * m3
	neg4 = x2[3*n:4*n, :, :, :] * m4

	loss = c_loss(neg1, neg2, neg3,  neg4, y, pos)
	return loss

class Sobelxy(nn.Module):
    def __init__(self, device):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device=device)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device=device)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx), torch.abs(sobely)

def grad(x1,x2,y):
	sobelconv = Sobelxy(device)
	vi_grad_x, vi_grad_y = sobelconv(x1)
	ir_grad_x, ir_grad_y = sobelconv(x2)
	fu_grad_x, fu_grad_y = sobelconv(y)
	grad_joint_x = torch.max(vi_grad_x, ir_grad_x)
	grad_joint_y = torch.max(vi_grad_y, ir_grad_y)
	loss_grad = F.l1_loss(grad_joint_x, fu_grad_x) + F.l1_loss(grad_joint_y, fu_grad_y)
	return loss_grad

def Fusion_loss(vi, ir, fu):
	loss_intensity=torch.mean(torch.pow((fu - vi), 2)) + torch.mean((vi < ir) * torch.abs((fu - ir)))
	return loss_intensity

