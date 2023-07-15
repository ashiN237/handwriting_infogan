import torch
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np
import random

def to_categorical(y, num_columns):
    y_cat = np.zeros((y.shape[0], num_columns))
    y = y.clip(0, num_columns - 1)  # インデックスが範囲外になるのを防ぐ
    y_cat[np.arange(y.shape[0]), y] = 1.0
    return y_cat

def sample_image(generator, static_z, static_label, static_code, opt, n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Static sample
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    if z.shape[0] < static_label.shape[0]:
        sampling_index = random.sample(range(0, static_label.shape[0]), z.shape[0])
        static_sample = generator(z, static_label[sampling_index], static_code[sampling_index])
    else:
        static_sample = generator(z, static_label, static_code)
    save_image(static_sample.data, opt.OUT_DIR + "static/%d.png" % batches_done, nrow=n_row, normalize=True)

    # Get varied c1 and c2
    zeros = np.zeros((n_row ** 2, 1))
    c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
    c1 = Variable(FloatTensor(np.concatenate((c_varied, zeros), -1)))
    c2 = Variable(FloatTensor(np.concatenate((zeros, c_varied), -1)))

    if c1.shape[0] < static_label.shape[0]:
        sample1 = generator(static_z[:c1.shape[0]], static_label[:c1.shape[0]], c1)
        sample2 = generator(static_z[:c1.shape[0]], static_label[:c1.shape[0]], c2)
    else:
        sample1 = generator(static_z, static_label[:c1.shape[0]], c1)
        sample2 = generator(static_z, static_label[:c1.shape[0]], c2)

    save_image(sample1.data, opt.OUT_DIR + "varying_c1/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(sample2.data, opt.OUT_DIR + "varying_c2/%d.png" % batches_done, nrow=n_row, normalize=True)
    
def save_model(generator, discriminator, epoch, history, model_path):
    torch.save(
        {
            "epoch": epoch,
            "history": history,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
        },
        model_path
    )
