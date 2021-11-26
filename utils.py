import os
import errno
from functools import reduce
import numpy as np

import torch.nn as nn
import torch
import torch.optim as optim
import torchvision.utils as vutils

from config import cfg


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
            
            
def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()
    
    
class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output
            

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


def build_optimizer(params, opt):
    if opt.optim == 'rmsprop':
        return optim.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, weight_decay=opt.weight_decay)
    elif opt.optim == 'adagrad':
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdm':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optim == 'adam':
        return optim.Adam(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def save_img_results(data_img, fake, epoch, image_dir):
    num = cfg.VIS_COUNT
    fake = fake[0:num]
    # data_img is changed to [0,1]
    if data_img is not None:
        data_img = data_img[0:num]
        vutils.save_image(
            data_img, '%s/real_samples.png' % image_dir,
            normalize=True)
        # fake.data is still [-1, 1]
        vutils.save_image(
            fake.data, '%s/fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)
    else:
        vutils.save_image(
            fake.data, '%s/lr_fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)


def decode_sequence(ix_to_word, seq):
    # N = seq.shape[0]
    # D = seq.shape[1]
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            # ix = seq[i,j+1]
            ix = seq[i, j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        out.append(txt)
    return out


def compute_discriminator_loss(netD, real_imgs, fake_imgs,real_labels, fake_labels,conditions):
    criterion = nn.BCELoss()
    batch_size = real_imgs.size(0)
    cond = conditions.detach()
    fake = fake_imgs.detach()
    real_features = netD(real_imgs)
    fake_features = netD(fake)
    # real pairs
    real_logits = netD.module.get_cond_logits(real_features, cond)
    errD_real = criterion(real_logits, real_labels)
    # wrong pairs
    wrong_logits = netD.module.get_cond_logits(real_features[:(batch_size-1)], cond[1:])
    errD_wrong = criterion(wrong_logits, fake_labels[1:])
    # fake pairs
    fake_logits = netD.module.get_cond_logits(fake_features, cond)
    errD_fake = criterion(fake_logits, fake_labels)

    if netD.module.get_uncond_logits is not None:
        real_logits = netD.module.get_uncond_logits(real_features)
        fake_logits = netD.module.get_uncond_logits(fake_features)
        uncond_errD_real = criterion(real_logits, real_labels)
        uncond_errD_fake = criterion(fake_logits, fake_labels)
        #
        errD = ((errD_real + uncond_errD_real) / 2. +
                (errD_fake + errD_wrong + uncond_errD_fake) / 3.)
        errD_real = (errD_real + uncond_errD_real) / 2.
        errD_fake = (errD_fake + uncond_errD_fake) / 2.
    else:
        errD = errD_real + (errD_fake + errD_wrong) * 0.5
    return errD, errD_real.item(), errD_wrong.item(), errD_fake.item()


def compute_generator_loss(netD, fake_imgs, real_labels, conditions):
    criterion = nn.BCELoss()
    cond = conditions.detach()
    fake_features = netD(fake_imgs)
    # fake pairs
    fake_logits = netD.module.get_cond_logits(fake_features, cond)
    errD_fake = criterion(fake_logits, real_labels)
    if netD.module.get_uncond_logits is not None:
        fake_logits = netD.module.get_uncond_logits(fake_features)
        uncond_errD_fake = criterion(fake_logits, real_labels)
        errD_fake += uncond_errD_fake
    return errD_fake


def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def save_img_results(data_img, fake, epoch, image_dir):
    num = cfg.VIS_COUNT
    fake = fake[0:num]
    # data_img is changed to [0,1]
    if data_img is not None:
        data_img = data_img[0:num]
        vutils.save_image(
            data_img, '%s/real_samples.png' % image_dir,
            normalize=True)
        # fake.data is still [-1, 1]
        vutils.save_image(
            fake.data, '%s/fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)
    else:
        vutils.save_image(
            fake.data, '%s/lr_fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)



def save_model(netG, netD, epoch, model_dir):
    torch.save(netG.state_dict(),'%s/netG_epoch_%d.pth' % (model_dir, epoch))
    torch.save(netD.state_dict(),'%s/netD_epoch_last.pth' % (model_dir))
    print('Save G/D models')