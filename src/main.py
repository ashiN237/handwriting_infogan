import argparse
import os
import numpy as np
import random
import itertools
import logging
import logging.handlers
import pickle
import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from pathlib import Path

from models import Generator, Discriminator, weights_init_normal
from data_loader import EtlCdbDataLoader
from utils import to_categorical
from train import train
from args import parse_args

torch.set_default_dtype(torch.float32)

# Set logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
rh = logging.handlers.RotatingFileHandler('./etlcdb_infogan.log', encoding='utf-8')
rh.setFormatter(logging.Formatter("%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s"))
logger.addHandler(rh)

if __name__ == "__main__":
    opt = parse_args()
    logger.debug(opt)

    cuda = True if torch.cuda.is_available() else False

    # Set seed
    seed = 123
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    opt.MODEL_DIR = 'jp_ckpt/'
    opt.OUT_DIR = 'jp_images/'
    os.makedirs(opt.OUT_DIR + "static/", exist_ok=True)
    os.makedirs(opt.OUT_DIR + "varying_c1/", exist_ok=True)
    os.makedirs(opt.OUT_DIR + "varying_c2/", exist_ok=True)
    os.makedirs(opt.MODEL_DIR, exist_ok=True)

    # Initialize generator and discriminator
    generator = Generator(opt)
    discriminator = Discriminator(opt)

    if cuda:
        generator.cuda()
        discriminator.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Loss functions
    adversarial_loss = torch.nn.MSELoss()
    categorical_loss = torch.nn.CrossEntropyLoss()
    continuous_loss = torch.nn.MSELoss()

    # Loss weights
    lambda_cat = 1
    lambda_con = 0.1

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_info = torch.optim.Adam(
        itertools.chain(generator.parameters(), discriminator.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
    )

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    # Static generator inputs for sampling
    static_z = Variable(FloatTensor(np.random.normal(0, 1, (opt.n_classes ** 2, opt.latent_dim))))
    static_label = to_categorical(
        np.array([num for _ in range(opt.n_classes) for num in range(opt.n_classes)]), num_columns=opt.n_classes
    )
    static_code = Variable(FloatTensor(np.random.uniform(-1, 1, (opt.n_classes ** 2, opt.code_dim))))

    def save_graph(history):
        plt.figure()
        plt.plot(range(0, opt.n_epochs), history['d_loss'], label='D loss')
        plt.plot(range(0, opt.n_epochs), history['g_loss'], label='G loss')
        plt.plot(range(0, opt.n_epochs), history['info_loss'], label='info loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig('jp_images/loss.png')

    # Configure data loader
    dataloader = torch.utils.data.DataLoader(
        EtlCdbDataLoader(
            "../ocr_dataset_create_jp/output",
            transform=transforms.Compose([
                transforms.Resize(opt.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        ),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        drop_last=True,
    )

    # ----------
    #  Training
    # ----------
    history = train(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        optimizer_info=optimizer_info,
        adversarial_loss=adversarial_loss,
        categorical_loss=categorical_loss,
        continuous_loss=continuous_loss,
        lambda_cat=lambda_cat,
        lambda_con=lambda_con,
        opt=opt,
        static_z=static_z,
        static_label=static_label,
        static_code=static_code
    )

    # --------------
    # Save Graph
    # --------------
    save_graph(history)
