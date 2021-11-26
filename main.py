import argparse
import os
import datetime, dateutil.tz
import pprint
import numpy as np
import random

import torchvision.transforms as transforms
import torch

from config import cfg, cfg_from_file
from dataloader import DataLoader
from trainer import Trainer
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='config.yml', type=str)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # set configuration parameters
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    print('learning_rate is {}'.format(cfg.learning_rate))

    ### set random seed
    setup_seed(cfg.seed)

    # name model with start time
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/{}_{}_{}'.format(cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    image_transform = transforms.Compose([
        transforms.RandomCrop(cfg.IMSIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    cfg.model_dir = os.path.join(output_dir, 'Model')
    cfg.image_dir = os.path.join(output_dir, 'Image')
    cfg.log_dir = os.path.join(output_dir, 'Log')
    cfg.log_dir_sc = os.path.join(output_dir, 'Log_sc')
    utils.mkdir_p(cfg.model_dir)
    utils.mkdir_p(cfg.image_dir)
    utils.mkdir_p(cfg.log_dir)
    utils.mkdir_p(cfg.log_dir_sc)

    # opt = dict(cfg)

    loader = DataLoader(cfg, transform=image_transform)
    Trainer(loader, cfg)





