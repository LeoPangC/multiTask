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
from model.diffusion.diffusion import GaussianDiffusion
from model.diffusion.unet import UNet
from DataProcess import GetWindSet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# from draw import draw_pics
import options as option
import utils as util
import math
from models import create_model


parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str,
                    default='/home/hy4080/PycharmProjects/multiTask/Exp/denoising-sde/options/train/ir-sde.yml',
                    help="Path to option YMAL file.")
parser.add_argument("--local_rank", type=int, default=0)
args = parser.parse_args()
opt = option.parse(args.opt, is_train=True)

opt = option.dict_to_nonedict(opt)
torch.backends.cudnn.benchmark = True
# 设置cuda:(cuda:0)
opt.device, = [torch.device("cuda:1" if torch.cuda.is_available() else "cpu")]

model = create_model(opt)  # load pretrained model of SFTMD
device = model.device

if __name__ == '__main__':
    opt.file = 'diff_i{}_e{}_{}_{}'.format(opt.n_timestep, opt.n_epochs, opt.linear_start, opt.linear_end)
    # opt.file = 'diff_i{}_e{}'.format(opt.n_timestep, opt.n_epochs)
    path = '/home/hy4080/PycharmProjects/multiTask'
    image_path = os.path.join(path, 'images', opt.file)
    save_path = os.path.join(path, 'save', opt.file)
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

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
        train_set = GetWindSet(gfs_data_1=gfsu_data[:9000],
                               era5_data_1=era5u_data[:9000],
                               gfs_data_2=gfsv_data[:9000],
                               era5_data_2=era5v_data[:9000])
        train_loader = DataLoader(train_set, batch_size=dataset_opt['batch_size'], shuffle=dataset_opt['use_shuffle'],
                                  drop_last=False,
                                  num_workers=dataset_opt['n_workers'])
        train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
        total_iters = int(opt["train"]["niter"])
        total_epochs = int(math.ceil(total_iters / train_size))

        for epoch in range(opt.n_epochs):

            train_epoch_loss = []
            for i, (dataX, dataY) in enumerate(train_dataloader):
                dataX = dataX.to(torch.float32).to(opt.device)
                dataY = dataY.to(torch.float32).to(opt.device)
                bias = dataY - dataX
                bias = bias.to(torch.float32).to(opt.device)

                loss = netG(dataX, bias)
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
            netG.load_state_dict(torch.load(os.path.join(save_path, 'diffusion_{}'.format(opt.n_epochs-1))))
            netG.eval()
            with torch.no_grad():
                print('第{}次测试'.format(i+1))
                img = netG.p_sample_loop(dataX, continous=True)

            bc_result = dataX + img
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
            save_image(img.data[:4], os.path.join(image_path, 'gt-bc_%d.png' % (i * opt.batch_size)), nrow=2,
                       normalize=True)
            save_image((dataY - dataX).data[:4], os.path.join(image_path, 'gt-gfs_%d.png' % (i * opt.batch_size)), nrow=2,
                       normalize=True)
