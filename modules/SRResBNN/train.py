import pdb
import torch
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import *
from modules.SRResBNN import Generator, Discriminator, vgg19, TVLoss, perceptual_loss
from logger.logger import info
import numpy as np
from PIL import Image


def train(args):
    transform = transforms.Compose([crop(args.scale, args.patch_size), augmentation()])
    train_dataset = DataExtractor(mode='train',
                                    lr_path=args.db_train_lr_path,
                                    hr_path=args.db_train_hr_path,
                                    transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    generator = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=args.res_num, scale=args.scale, args=args)
    if args.fine_tuning:
        info('Fine-tuning is running')
        save_dir = f'modules/SRResBNN/models/SRGAN'
        generator.load_state_dict(torch.load(args.generator_path))
        info("Pre-trained model has been loaded")
        info("Path: %s" % (args.generator_path))
    else:
        info('Pre-training is running')
        save_dir = f'modules/SRResBNN/models/SRResBNN'
    os.makedirs(save_dir, exist_ok=True)


    generator = generator.to(device)
    generator.train()

    l2_loss = nn.MSELoss()
    g_optim = optim.Adam(generator.parameters(), lr=1e-4)
    pre_epoch = 0
    fine_epoch = 0
    loss_hist = []
    #### Train using L2_loss
    while pre_epoch < args.pre_train_epoch:
        iter_loss = []
        for i, tr_data in enumerate(train_loader):
            hr = tr_data['hr'].type(torch.cuda.FloatTensor).to(device)
            lr = tr_data['lr'].type(torch.cuda.FloatTensor).to(device)

            output, _ = generator(lr)
            loss = l2_loss(hr, output) + args.lambd*torch.norm(generator.conv02.body[0].weight.data, p=2)

            g_optim.zero_grad()
            loss.backward()
            g_optim.step()
            iter_loss.append(loss.item())

        loss_hist.append(np.average(iter_loss))
        info(f'Epoch: {pre_epoch} | Loss: {loss_hist[-1]:.7f}')
        pre_epoch += 1


        if pre_epoch % 100 == 0:
            np.save(os.path.join(save_dir, 'train_loss.npy'), np.array(loss_hist))
            torch.save(generator.state_dict(), os.path.join(save_dir, 'SRResBNN_%03d.pt' % pre_epoch))

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
        info(f'SRResBNN Epoch: {fine_epoch} | Generator loss: {g_loss.item():.7f} | Discriminator loss: {d_loss.item():.7f}')

        if fine_epoch % 100 == 0:
            np.save(os.path.join(save_dir, 'generator_loss.npy'), np.array(generator_loss_hist))
            np.save(os.path.join(save_dir, 'discriminator_loss.npy'), np.array(discriminator_loss_hist))
            plt.plot(np.array(generator_loss_hist), label='Generator')
            plt.plot(np.array(generator_loss_hist), label='Discriminator')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('SRResBNN')
            plt.savefig(os.path.join(save_dir, 'SRResBNN_loss.png'), dpi=250)
            plt.close()
            torch.save(generator.state_dict(), os.path.join(save_dir, 'SRResBNN_gene_%03d.pt' % fine_epoch))
            torch.save(discriminator.state_dict(), os.path.join(save_dir, 'SRResBNN_discrim_%03d.pt' % fine_epoch))
    