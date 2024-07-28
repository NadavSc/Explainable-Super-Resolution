import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import *
from modules.SRGAN import Generator, Discriminator, vgg19, TVLoss, perceptual_loss
from logger.logger import info
import numpy as np
from PIL import Image


def train(args, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    generator = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=args.res_num, scale=args.scale)
    if args.fine_tuning:
        generator.load_state_dict(torch.load(args.generator_path))
        info("Pre-trained model has been loaded")
        info("Path: %s" % (args.generator_path))

    generator = generator.to(device)
    generator.train()

    l2_loss = nn.MSELoss()
    g_optim = optim.Adam(generator.parameters(), lr=1e-4)
    pre_epoch = 0
    fine_epoch = 0

    #### Train using L2_loss
    while pre_epoch < args.pre_train_epoch:
        for i, tr_data in enumerate(train_loader):
            hr = tr_data['hr'].type(torch.cuda.FloatTensor).to(device)
            lr = tr_data['lr'].type(torch.cuda.FloatTensor).to(device)

            output, _ = generator(lr)
            loss = l2_loss(hr, output)

            g_optim.zero_grad()
            loss.backward()
            g_optim.step()

        pre_epoch += 1

        if pre_epoch % 2 == 0:
            print(pre_epoch)
            print(loss.item())
            print('=========')

        if pre_epoch % 800 == 0:
            torch.save(generator.state_dict(), './models/pre_trained_model_%03d.pt' % pre_epoch)

    #### Train using perceptual & adversarial loss
    vgg_net = vgg19().to(device)
    vgg_net = vgg_net.eval()

    discriminator = Discriminator(patch_size=args.patch_size * args.scale)
    discriminator = discriminator.to(device)
    discriminator.train()

    d_optim = optim.Adam(discriminator.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(g_optim, step_size=2000, gamma=0.1)

    VGG_loss = perceptual_loss(vgg_net)
    cross_ent = nn.BCELoss()
    tv_loss = TVLoss()
    generator_loss_hist = []
    discriminator_loss_hist = []
    while fine_epoch < args.fine_train_epoch:

        scheduler.step()

        for i, tr_data in enumerate(train_loader):
            hr = tr_data['hr'].type(torch.cuda.FloatTensor).to(device)
            lr = tr_data['lr'].type(torch.cuda.FloatTensor).to(device)

            ## Training Discriminator
            output, _ = generator(lr)
            fake_prob = discriminator(output)
            real_prob = discriminator(hr)

            fake_label = torch.zeros((len(fake_prob), 1)).to(device)
            real_label = torch.ones((len(real_prob), 1)).to(device)

            d_loss_real = cross_ent(real_prob, real_label)
            d_loss_fake = cross_ent(fake_prob, fake_label)

            d_loss = d_loss_real + d_loss_fake

            g_optim.zero_grad()
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            ## Training Generator
            output, _ = generator(lr)
            fake_prob = discriminator(output)

            _percep_loss, hr_feat, sr_feat = VGG_loss((hr + 1.0) / 2.0, (output + 1.0) / 2.0, layer=args.feat_layer)

            L2_loss = l2_loss(output, hr)
            percep_loss = args.vgg_rescale_coeff * _percep_loss
            adversarial_loss = args.adv_coeff * cross_ent(fake_prob, real_label)
            total_variance_loss = args.tv_loss_coeff * tv_loss(args.vgg_rescale_coeff * (hr_feat - sr_feat) ** 2)

            g_loss = percep_loss + adversarial_loss + total_variance_loss + L2_loss

            g_optim.zero_grad()
            d_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

        fine_epoch += 1
        generator_loss_hist.append(g_loss.item())
        discriminator_loss_hist.append(d_loss.item())
        import pdb; pdb.set_trace()
        info(f'SRGAN Epoch: {fine_epoch} | Generator loss: {g_loss.item():.7f} | Discriminator loss: {d_loss.item():.7f}')

        if fine_epoch % 100 == 0:
            np.save('./models/generator_loss.npy' , np.array(generator_loss_hist))
            np.save('./models/discriminator_loss.npy' , np.array(generator_loss_hist))
            torch.save(generator.state_dict(), './models/SRGAN_gene_%03d.pt' % fine_epoch)
            torch.save(discriminator.state_dict(), './models/SRGAN_discrim_%03d.pt' % fine_epoch)
