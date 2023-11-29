import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch
from model.Unet import DownsampleLayer, UpSampleLayer
from Decoder import Decoder
from Encoder import Encoder
from DataProcess import WindSet, GetWindSet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--D_lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--G_lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
parser.add_argument("--dim", type=int, default=1024, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=96, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=2, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval betwen image samples")
parser.add_argument("--heads", type=int, default=8)
parser.add_argument("--patch_size", type=int, default=8)
parser.add_argument("--mlp_dim", type=int, default=2048)
parser.add_argument("--dim_head", type=int, default=64)
parser.add_argument("--D_depth", type=int, default=2)
parser.add_argument("--G_depth", type=int, default=4)
parser.add_argument("--dropout", type=float, default=0.)
parser.add_argument("--emb_dropout", type=float, default=0.)
parser.add_argument("--device", type=str, default="cpu")
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)
img_area = np.prod(img_shape)

# 设置cuda:(cuda:0)
opt.device, = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu")]

# u10_gfs = np.load('Dataset/u10_gfs.npy')
# v10_gfs = np.load('Dataset/v10_gfs.npy')
# u10_gfs = np.expand_dims(u10_gfs, axis=1)
# v10_gfs = np.expand_dims(v10_gfs, axis=1)
# wind_gfs = np.concatenate((u10_gfs, v10_gfs), axis=1)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        out_channels = [2**(i+6) for i in range(5)]  # [64, 128, 256, 512, 1024]
        # 下采样
        self.du_1 = DownsampleLayer(1, out_channels[0])
        self.du_2 = DownsampleLayer(out_channels[0], out_channels[1])
        self.du_3 = DownsampleLayer(out_channels[1], out_channels[2])
        self.du_4 = DownsampleLayer(out_channels[2], out_channels[3])
        self.dv_1 = DownsampleLayer(1, out_channels[0])
        self.dv_2 = DownsampleLayer(out_channels[0], out_channels[1])
        self.dv_3 = DownsampleLayer(out_channels[1], out_channels[2])
        self.dv_4 = DownsampleLayer(out_channels[2], out_channels[3])
        # 上采样
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels[4], out_channels=out_channels[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU()
        )
        self.u1 = UpSampleLayer(out_channels[3], out_channels[3])  # 512-1024-521
        self.u2 = UpSampleLayer(out_channels[4], out_channels[2])  # 1024-512-256
        self.u3 = UpSampleLayer(out_channels[3], out_channels[1])  # 512-256-128
        self.u4 = UpSampleLayer(out_channels[2], out_channels[0])  # 256-128-64
        # 输出
        self.o = nn.Sequential(
            nn.Conv2d(out_channels[1], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], opt.channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        out_1_u, out1_u = self.du_1(x[:, 0:1])
        out_1_v, out1_v = self.dv_1(x[:, 1:2])
        out_2_u, out2_u = self.du_2(out1_u)
        out_2_v, out2_v = self.dv_2(out1_v)
        out_3_u, out3_u = self.du_3(out2_u)
        out_3_v, out3_v = self.dv_3(out2_v)
        out_4_u, out4_u = self.du_4(out3_u)
        out_4_v, out4_v = self.dv_4(out3_v)
        out4 = torch.cat((out4_u, out4_v), dim=1)
        out4 = self.Conv_BN_ReLU_2(out4)
        out5 = self.u1(out4, out_4_u, out_4_v)
        out6 = self.u2(out5, out_3_u, out_3_v)
        out7 = self.u3(out6, out_2_u, out_2_v)
        out8 = self.u4(out7, out_1_u, out_1_v)
        out = self.o(out8)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.decoder = Decoder(image_size=opt.img_size,
                               patch_size=opt.patch_size,
                               dim=opt.dim,
                               depth=opt.D_depth,
                               heads=opt.heads,
                               mlp_dim=opt.mlp_dim,
                               channels=opt.channels,
                               dim_head=opt.dim_head,
                               dropout=opt.dropout,
                               emb_dropout=opt.emb_dropout)

    def forward(self, z):
        validity = self.decoder(z)
        return validity


# transformer = Transformer().to(opt.device)
# 生成器判别器
generator = Generator().to(opt.device)
# discriminator = Discriminator().to(opt.device)

criterion = torch.nn.MSELoss(reduction='mean')
# 定义分类损失函数
# criterion_b = nn.BCELoss()
# 其次定义 优化函数,优化函数的学习率为0.0003
# betas:用于计算梯度以及梯度平方的运行平均值的系数
# optimizer = torch.optim.Adam(transformer.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.G_lr, betas=(opt.b1, opt.b2))
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.D_lr, betas=(opt.b1, opt.b2))

if torch.cuda.is_available():
    # transformer = transformer.cuda()
    generator = generator.cuda()
    # discriminator = discriminator.cuda()
    criterion = criterion.cuda()
    # criterion_b = criterion_b.cuda()


train_loss = []
valid_loss = []
D_train_epochs_loss = []
G_train_epochs_loss = []
valid_epochs_loss = []

# ----------
#  Training
# ----------
# 进行多个epoch的训练
era5u_path = 'era5u_train.npy'
gfsu_path = 'gfsu_train.npy'
era5v_path = 'era5v_train.npy'
gfsv_path = 'gfsv_train.npy'
path = '/media/upc/40B0E99FB0E99C22/code/multiTask/Dataset'
gfsu_data = np.load(os.path.join(path, gfsu_path))[:, :96, :96]
era5u_data = np.load(os.path.join(path, era5u_path))[:, :96, :96]
gfsv_data = np.load(os.path.join(path, gfsv_path))[:, :96, :96]
era5v_data = np.load(os.path.join(path, era5v_path))[:, :96, :96]

deg = 180.0 / np.pi
era5_speed = np.sqrt(era5u_data ** 2 + era5v_data ** 2)
gfs_speed = np.sqrt(gfsu_data ** 2 + gfsv_data ** 2)
era5_dir = 180.0 + np.arctan2(era5u_data, era5v_data) * deg
gfs_dir = 180.0 + np.arctan2(gfsu_data, gfsv_data) * deg

s_max = 64.817894
era5_speed = era5_speed / s_max
gfs_speed = gfs_speed / s_max
era5_dir = era5_dir / 360.0
gfs_dir = gfs_dir / 360.0
wind_train = GetWindSet(gfs_data_1=gfs_speed[:9000], era5_data_1=era5_speed[:9000], gfs_data_2=gfs_dir[:9000], era5_data_2=era5_dir[:9000])
train_dataloader = DataLoader(wind_train, batch_size=opt.batch_size, shuffle=False, drop_last=False, num_workers=2)
wind_valid = GetWindSet(gfs_data_1=gfs_speed[9000: 500], era5_data_1=era5_speed[9000: 500], gfs_data_2=gfs_dir[9000: 500], era5_data_2=era5_dir[9000: 500])
valid_dataloader = DataLoader(wind_valid, batch_size=opt.batch_size, shuffle=False, drop_last=False)

for epoch in range(opt.n_epochs):  # epoch:50
    # transformer.train()
    # train_epoch_loss = []
    generator.train()
    # discriminator.train()
    D_train_epoch_loss = []
    G_train_epoch_loss = []
    for i, (dataX, dataY) in enumerate(train_dataloader):  # dataX:(64, 1, 180, 180) dataY:(64, 1, 180, 180)
        # dataX = (dataX - gfs_min) / (gfs_max - gfs_min)
        # dataY = (dataY - era5_min) / (era5_max - era5_min)
        # =============================训练判别器==================
        # real_label = torch.ones(dataY.size(0), 1).to(opt.device) # 定义真实的图片label为1
        # fake_label = torch.zeros(dataX.size(0), 1).to(opt.device)  # 定义假的图片的label为0
        dataX = dataX.to(torch.float32).to(opt.device)
        dataY = dataY.to(torch.float32).to(opt.device)

        # ---------------------
        # Train Discriminator
        # 分为两部分：1、真的图像判别为真；2、假的图像判别为假
        # ---------------------
        # 计算真实图片的损失
        # real_out = discriminator(dataY)  # 将真实图片放入判别器中
        # loss_real_D = criterion_b(real_out, real_label)  # 得到真实图片的loss
        # real_scores = real_out  # 得到真实图片的判别值，输出的值越接近1越好
        # 计算假的图片的损失
        # fake_img = generator(dataX).detach()  # 随机噪声放入生成网络中，生成一张假的图片。
        # fake_out = discriminator(fake_img)  # 判别器判断假的图片
        # loss_fake_D = criterion_b(fake_out, fake_label)  # 得到假的图片的loss
        # fake_scores = fake_out  # 得到假图片的判别值，对于判别器来说，假图片的损失越接近0越好
        # 损失函数和优化
        # loss_D = loss_real_D + loss_fake_D  # 损失包括判真损失和判假损失
        # optimizer_D.zero_grad()  # 在反向传播之前，先将梯度归0
        # loss_D.backward()  # 将误差反向传播
        # optimizer_D.step()  # 更新参数

        # -----------------
        # Train Generator
        # 原理：目的是希望生成的假的图片被判别器判断为真的图片，
        # 在此过程中，将判别器固定，将假的图片传入判别器的结果与真实的label对应，
        # 反向传播更新的参数是生成网络里面的参数，
        # 这样可以通过更新生成网络里面的参数，来训练网络，使得生成的图片让判别器以为是真的, 这样就达到了对抗的目的
        # -----------------
        output = generator(dataX)  # 随机噪声输入到生成器中，得到一副假的图片
        # output = discriminator(fake_img)  # 经过判别器得到的结果
        # 损失函数和优化
        loss = criterion(output, dataY)  # 得到的假的图片与真实的图片的label的loss
        optimizer_G.zero_grad()  # 梯度归0
        loss.backward()  # 进行反向传播
        optimizer_G.step()  # step()一般用在反向传播后面,用于更新生成网络的参数

        # outputs = transformer(dataX)
        # optimizer.zero_grad()
        # loss = criterion(dataY, outputs)
        # loss.backward()
        # optimizer.step()
        # D_train_epoch_loss.append(loss_D.item())
        G_train_epoch_loss.append(loss.item())
        # train_loss.append(loss.item())
        if i % (len(train_dataloader)//2) == 0:
            print("epoch={}/{},{}/{}of train, loss={}".format(
                epoch, opt.n_epochs, i, len(train_dataloader), loss.item()))
        # 保存训练过程中的图像
        batches_done = epoch * len(train_dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(output.data[:25], "../images/unet_sd/%d.png" % batches_done, nrow=5, normalize=True)
    # D_train_epochs_loss.append(np.average(D_train_epoch_loss))
    G_train_epochs_loss.append(np.average(G_train_epoch_loss))

    # =====================valid============================
    # transformer.eval()
    # valid_epoch_loss = []
    # for idx, (dataX, dataY) in enumerate(u10_valid_dataloader, 0):
    #     # dataX = (dataX - gfs_min) / (gfs_max - gfs_min)
    #     # dataY = (dataY - era5_min) / (era5_max - era5_min)
    #     dataX = dataX.to(torch.float32).to(opt.device)
    #     dataY = dataY.to(torch.float32).to(opt.device)
    #     outputs = transformer(dataX)
    #     loss = criterion(dataY, outputs)
    #     valid_epoch_loss.append(loss.item())
    #     valid_loss.append(loss.item())
    # valid_epochs_loss.append(np.average(valid_epoch_loss))

    # ====================adjust lr========================
    # D_lr_adjust = {
    #     2: 1e-5, 4: 5e-6, 6: 1e-6, 8: 5e-7,
    #     10: 1e-7, 15: 5e-8, 20: 1e-8
    # }
    # G_lr_adjust = {
    #     2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
    #     10: 5e-7, 15: 1e-7, 20: 5e-8
    # }
    # if epoch in D_lr_adjust.keys():
    #     lr = D_lr_adjust[epoch]
    #     for param_group in optimizer_D.param_groups:
    #         param_group['lr'] = lr
    # if epoch in G_lr_adjust.keys():
    #     lr = G_lr_adjust[epoch]
    #     for param_group in optimizer_G.param_groups:
    #         param_group['lr'] = lr
    #     print('Updating learning rate to {}'.format(lr))
    # torch.save(transformer.state_dict(), './save/gan/wind_u_{}.pth'.format(epoch))
    if epoch % 10 == 0:
        torch.save(generator.state_dict(), '../save/unet_sd/unet_sd_{}.pth'.format(epoch))
    # torch.save(discriminator.state_dict(), './save/gan/discriminator_u_{}.pth'.format(epoch))


plt.figure(figsize=(12, 4))
# plt.subplot(121)
plt.plot(train_loss[:])
plt.title("train_loss")
# plt.subplot(122)
# plt.plot(D_train_epochs_loss[1:], '--', label="D train_loss")
plt.plot(G_train_epochs_loss[1:], '--', label="G train_loss")
# plt.plot(valid_epochs_loss[1:], '--', label="valid_loss")
plt.title("epochs_loss")
plt.legend()
plt.show()