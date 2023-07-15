from torch.cuda import FloatTensor, LongTensor
from torch.autograd import Variable
import torch
import numpy as np
import itertools
import os

from utils import to_categorical, sample_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(epoch, history, model_path):
    torch.save(
        {
            "epoch": epoch,
            "history": history,
        },
        model_path
    )

def train(generator, discriminator, dataloader, optimizer_G, optimizer_D, optimizer_info,
          adversarial_loss, categorical_loss, continuous_loss, lambda_cat, lambda_con, opt,
          static_z, static_label, static_code):
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

    best_loss = 10

    history = {
        "d_loss": [],
        "g_loss": [],
        "info_loss": []
    }

    epoch_start = getattr(opt, 'epoch_start', 0)  # opt.epoch_start の値を取得します。存在しない場合は 0 を使用します。

    for epoch in range(epoch_start, opt.n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = to_categorical(labels.numpy(), num_columns=opt.n_classes)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            label_input = torch.from_numpy(to_categorical(np.random.randint(0, opt.n_classes, batch_size), num_columns=opt.n_classes)).type(FloatTensor)
            code_input = torch.from_numpy(np.random.uniform(-1, 1, (batch_size, opt.code_dim))).type(FloatTensor)

            # Generate a batch of images
            gen_imgs = generator(z, label_input, code_input)


            # Loss measures generator's ability to fool the discriminator
            validity, _, _ = discriminator(gen_imgs)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            real_pred, _, _ = discriminator(real_imgs)
            d_real_loss = adversarial_loss(real_pred, valid)

            # Loss for fake images
            fake_pred, _, _ = discriminator(gen_imgs.detach())
            d_fake_loss = adversarial_loss(fake_pred, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # ------------------
            # Information Loss
            # ------------------

            optimizer_info.zero_grad()

            # Sample labels
            sampled_labels = np.random.randint(0, opt.n_classes, batch_size)

            # Ground truth labels
            gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False)

            # Sample noise, labels and code as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            label_input = to_categorical(sampled_labels, num_columns=opt.n_classes)
            code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

            gen_imgs = generator(z, label_input, code_input)
            _, pred_label, pred_code = discriminator(gen_imgs)

            info_loss = lambda_cat * categorical_loss(pred_label, gt_labels) + lambda_con * continuous_loss(
                pred_code, code_input
            )

            info_loss.backward()
            optimizer_info.step()

            # --------------
            # Save best_loss
            # --------------
            if info_loss.item() < best_loss:
                best_loss = info_loss.item()
                torch.save(
                    {
                        "epoch": epoch,
                        "history": history,
                        "generator": generator.state_dict(),
                        "discriminator": discriminator.state_dict(),
                    },
                    opt.MODEL_DIR + "jp_best_model_%d.tar" % epoch
                )
                print(
                    "Save best model [Model Path %s][Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
                    % (opt.MODEL_DIR + "jp_best_model_%d.tar" % epoch, epoch, opt.n_epochs, i, len(dataloader),
                       d_loss.item(), g_loss.item(), info_loss.item())
                )
            # --------------
            # Log Progress
            # --------------
            if i % len(dataloader) == 0:
                # append loss
                if epoch < len(history['d_loss']):
                    history['d_loss'][epoch] = d_loss.item()
                    history['g_loss'][epoch] = g_loss.item()
                    history['info_loss'][epoch] = info_loss.item()
                else:
                    history['d_loss'].append(d_loss.item())
                    history['g_loss'].append(g_loss.item())
                    history['info_loss'].append(info_loss.item())

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), info_loss.item())
                )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                sample_image(generator, static_z, static_label, static_code, opt, n_row=10, batches_done=batches_done)
                print('sample image done batches_done: %d' % batches_done)

        # --------------
        # Save model
        # --------------
        if epoch % 20 == 0 or epoch == (opt.n_epochs - 1):
            save_model(epoch, history, opt.MODEL_DIR + "jp_model_%d.tar" % epoch)
            print(f"Save model [Epoch {epoch}/{opt.n_epochs}] [Path jp_ckpt/jp_model_{epoch}.tar]")

    return history
