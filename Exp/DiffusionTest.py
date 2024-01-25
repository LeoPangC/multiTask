import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch
from torch.nn import init
from model.sr3_modules.diffusion import GaussianDiffusion
from model.sr3_modules.unet import UNet
from DataProcess import GetWindSet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
# from draw import draw_pics


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
parser.add_argument("--dim", type=int, default=1024, help="dimensionality of the latent space")
parser.add_argument("--image_size", type=int, default=96, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=2, help="number of image channels")
parser.add_argument("--linear_start", type=float, default=2e-6)
# 上次区间值为 1e-2-1e-6
parser.add_argument("--linear_end", type=float, default=2e-2)
parser.add_argument("--conditional", type=bool, default=True)
parser.add_argument("--n_iter", type=int, default=1000000)

parser.add_argument("--in_channel", type=int, default=4)
parser.add_argument("--out_channel", type=int, default=2)
parser.add_argument("--inner_channel", type=int, default=32)
parser.add_argument("--norm_groups", type=int, default=32)
parser.add_argument("--channel_mults", type=tuple, default=(1, 2, 4, 8, 8))
parser.add_argument("--attn_res", type=tuple, default=(6,))
parser.add_argument("--res_blocks", type=int, default=3)
parser.add_argument("--with_time_emb", type=bool, default=True)

parser.add_argument("--dropout", type=float, default=0.)
parser.add_argument("--device", type=str, default="cpu")

parser.add_argument("--schedule", type=str, default='linear')
parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs of training")
parser.add_argument("--mode", type=str, default='train', help="[train, test]")
parser.add_argument("--n_timestep", type=int, default=4000)
parser.add_argument("--file", type=str, default='diff_i2000')
opt = parser.parse_args()
# 设置cuda:(cuda:0)
opt.device, = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu")]

model = UNet(
    in_channel=opt.in_channel,
    out_channel=opt.out_channel,
    norm_groups=opt.norm_groups,
    inner_channel=opt.inner_channel,
    channel_mults=opt.channel_mults,
    attn_res=opt.attn_res,
    res_blocks=opt.res_blocks,
    dropout=opt.dropout,
    image_size=opt.image_size
)

netG = GaussianDiffusion(
    model,
    image_size=opt.image_size,
    channels=opt.channels,
    loss_type='l1',  # L1 or L2
    conditional=opt.conditional,
    schedule_opt={
        'schedule': opt.schedule,
        'n_timestep': opt.n_timestep,
        'linear_start': opt.linear_start,
        'linear_end': opt.linear_end
    },
    device=opt.device
)

if __name__ == '__main__':
    opt.file = 'diff_i{}_e{}_{}_{}'.format(opt.n_timestep, opt.n_epochs, opt.linear_start, opt.linear_end)
    # opt.file = 'diff_i{}_e{}'.format(opt.n_timestep, opt.n_epochs)
    path = '/home/hy4080/PycharmProjects/multiTask'
    image_path = os.path.join(path, 'images', opt.file)
    save_path = os.path.join(path, 'save', opt.file)
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    netG = netG.to(opt.device)

    netG.train()
    optim_params = list(netG.parameters())

    optimizer = torch.optim.Adam(optim_params, lr=opt.lr)
    if torch.cuda.is_available():
        generator = netG.cuda()

    train_loss = []
    valid_loss = []
    train_epochs_loss = []
    valid_epochs_loss = []

    era5u_path = 'era5u_train.npy'
    gfsu_path = 'gfsu_train.npy'
    era5v_path = 'era5v_train.npy'
    gfsv_path = 'gfsv_train.npy'
    data_path = '/home/hy4080/PycharmProjects/multiTask/Dataset/'
    gfsu_data = np.load(os.path.join(data_path, gfsu_path))[:, :96, :96]
    era5u_data = np.load(os.path.join(data_path, era5u_path))[:, :96, :96]
    gfsv_data = np.load(os.path.join(data_path, gfsv_path))[:, :96, :96]
    era5v_data = np.load(os.path.join(data_path, era5v_path))[:, :96, :96]
    #  u: max:49.1 min:-55.7
    #  v: max:60.6 min:-54.8
    # gfsu_data = (gfsu_data + 55.7) / (49.1 + 55.7) * 2 - 1
    # era5u_data = (era5u_data + 55.7) / (49.1 + 55.7) * 2 - 1
    # gfsv_data = (gfsv_data + 54.8) / (60.6 + 54.8) * 2 - 1
    # era5v_data = (era5v_data + 54.8) / (60.6 + 54.8) * 2 - 1
    if opt.mode == 'train':
        def weights_init_orthogonal(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                init.orthogonal_(m.weight.data, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('Linear') != -1:
                init.orthogonal_(m.weight.data, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('BatchNorm2d') != -1:
                init.orthogonal_(m.weight.data, 1.0)
                init.orthogonal_(m.bias.data, 0.0)
        netG.apply(weights_init_orthogonal)
        wind_train = GetWindSet(gfs_data_1=gfsu_data[:9000], era5_data_1=era5u_data[:9000], gfs_data_2=gfsv_data[:9000],
                                era5_data_2=era5v_data[:9000])
        train_dataloader = DataLoader(wind_train, batch_size=opt.batch_size, shuffle=False, drop_last=False, num_workers=2)

        for epoch in range(opt.n_epochs):

            train_epoch_loss = []
            for i, (dataX, dataY) in enumerate(train_dataloader):
                dataX = dataX.to(torch.float32).to(opt.device)
                dataY = dataY.to(torch.float32).to(opt.device)
                # lr = F.resize(dataX, [24, 24])
                # sr = F.resize(lr, [96, 96])
                # sr = sr.to(torch.float32).to(opt.device)
                # bias = dataY - dataX
                # bias = bias.to(torch.float32).to(opt.device)

                # loss = netG(dataX, bias)
                loss = netG(dataX, dataY)
                b, c, h, w = dataX.shape
                loss = loss.sum() / (b*c*h*w)
                optimizer.zero_grad()  # 梯度归0
                loss.backward()  # 进行反向传播
                optimizer.step()  # step()一般用在反向传播后面,用于更新生成网络的参数

                train_epoch_loss.append(loss.item())
                if i % (len(train_dataloader)//2) == 0:
                    print("epoch={}/{},{}/{}of train, loss={}".format(
                        epoch, opt.n_epochs, i, len(train_dataloader), loss.item()))

            if (epoch + 1) % 10 == 0 and epoch >= 10:
                state_dict = netG.state_dict()
                for key, param in state_dict.items():
                    state_dict[key] = param.cpu()
                torch.save(state_dict, os.path.join(save_path, 'diffusion_{}'.format(opt.n_epochs-1)))

            train_epochs_loss.append(np.average(train_epoch_loss))

        plt.figure(figsize=(12, 4))
        # plt.subplot(121)
        plt.plot(train_loss[:])
        plt.title("train_loss")
        # plt.subplot(122)
        # plt.plot(D_train_epochs_loss[1:], '--', label="D train_loss")
        plt.plot(train_epochs_loss[1:], '--', label="G train_loss")
        # plt.plot(valid_epochs_loss[1:], '--', label="valid_loss")
        plt.title("epochs_loss")
        plt.legend()
        plt.savefig(os.path.join(image_path, 'loss.png'))

    elif opt.mode == 'test':
        wind_test = GetWindSet(gfs_data_1=gfsu_data[9000:], era5_data_1=era5u_data[9000:], gfs_data_2=gfsv_data[9000:],
                                era5_data_2=era5v_data[9000:])
        test_dataloader = DataLoader(wind_test, batch_size=opt.batch_size, shuffle=False, drop_last=False, num_workers=2)
        print('test beginning')
        for i, (dataX, dataY) in enumerate(test_dataloader):
            b, c, h, w = dataX.shape
            dataX = dataX.to(torch.float32).to(opt.device)
            dataY = dataY.to(torch.float32).to(opt.device)
            # lr = F.resize(dataX, [24, 24])
            # lr_c = F.center_crop(lr, [24, 24])
            # hr = F.resize(dataX[:4], [96, 96])
            # hr_c = F.center_crop(hr, [96, 96])
            # sr = F.resize(lr, [96, 96])
            # sr = sr.to(torch.float32).to(opt.device)
            # sr_c = F.center_crop(sr, [96, 96])
            # save_image(lr.data[:4], os.path.join(image_path, 'lr_%d.png' % (i * opt.batch_size)), nrow=2,
            #            normalize=True)
            # save_image(lr_c.data[:4], os.path.join(image_path, 'lr_c_%d.png' % (i * opt.batch_size)), nrow=2,
            #            normalize=True)
            # save_image(hr.data[:4], os.path.join(image_path, 'hr_%d.png' % (i * opt.batch_size)), nrow=2,
            #            normalize=True)
            # save_image(hr_c.data[:4], os.path.join(image_path, 'hr_c_%d.png' % (i * opt.batch_size)), nrow=2,
            #            normalize=True)
            # save_image(sr.data[:4], os.path.join(image_path, 'sr_%d.png' % (i * opt.batch_size)), nrow=2,
            #            normalize=True)
            # save_image(sr_c.data[:4], os.path.join(image_path, 'sr_c_%d.png' % (i * opt.batch_size)), nrow=2,
            #            normalize=True)
            # netG.load_state_dict(torch.load(os.path.join(save_path, 'diffusion_{}'.format(opt.n_epochs-1))))
            netG.eval()
            with torch.no_grad():
                print('第{}次测试'.format(i+1))
                img = netG.p_sample_loop(dataX, continous=True)

            # bc_result = dataX + img
            bc_result = img
            origin = dataX.cpu().numpy()
            predict_result = bc_result.cpu().numpy()
            label = dataY.cpu().numpy()
            ori_rmse = np.sqrt(np.sum((origin-label)**2)/(b*c*h*w))
            bc_rmse = np.sqrt(np.sum((predict_result-label)**2)/(b*c*h*w))
            u10_rmse = np.sqrt(np.sum((origin[:, 0]-label[:, 0])**2)/(b*h*w))
            v10_rmse = np.sqrt(np.sum((origin[:, 1]-label[:, 1])**2)/(b*h*w))
            u10_bc_rmse = np.sqrt(np.sum((predict_result[:, 0]-label[:, 0])**2)/(b*h*w))
            v10_bc_rmse = np.sqrt(np.sum((predict_result[:, 1]-label[:, 1])**2)/(b*h*w))
            print('u10 rmse:', u10_rmse)
            print('v10 rmse:', v10_rmse)
            print('origin rmse:', ori_rmse)
            print('u10 bc rmse:', u10_bc_rmse)
            print('v10 bc rmse:', v10_bc_rmse)
            print('bc rmse:', bc_rmse)
            print('u10:', (u10_rmse - u10_bc_rmse)/u10_rmse)
            print('v10:', (v10_rmse - v10_bc_rmse)/v10_rmse)
            print('result:', (ori_rmse - bc_rmse)/ori_rmse)

            save_image(dataX.data[:4], os.path.join(image_path, 'gfs_%d.png' % (i * opt.batch_size)), nrow=2,
                       normalize=True)
            save_image(bc_result.data[:4], os.path.join(image_path, 'bc_%d.png' % (i * opt.batch_size)), nrow=2, normalize=True)
            save_image(dataY.data[:4], os.path.join(image_path, 'gt_%d.png' % (i * opt.batch_size)), nrow=2, normalize=True)
            save_image((dataY - bc_result).data[:4], os.path.join(image_path, 'gt-bc_%d.png' % (i * opt.batch_size)), nrow=2,
                       normalize=True)
            save_image((dataY - bc_result).data[:4], os.path.join(image_path, 'gt-gfs_%d.png' % (i * opt.batch_size)), nrow=2,
                       normalize=True)
