import torch
from layers import HyperbolicVAE, VanillaVAE
from branching_process import BranchingLoaders
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim

import utils
import os
from utils import ROOT_DIR, CSV_DIR, MODELS_DIR, IMAGES_DIR
from config import config
import pandas as pd
import helpers
import traceback


def train(model, opt, train_loader, epoch, args, df=None):
    model.train()
    mu_loss = 0
    mu_recon = 0
    mu_kl_div = 0

    for ix, (xsample, label) in enumerate(train_loader):

        if args.dataset in ['mnist']:
            # Flatten
            xsample = xsample.view(xsample.size(0), -1)

        opt.zero_grad()

        recon, kl_div = model.loss_fn(xsample)

        loss = recon + args.beta * kl_div
        loss.backward()

        mu_loss += loss.item()
        mu_recon += recon.item()
        mu_kl_div += kl_div.item()

        opt.step()

        if args.break_early == 1 and (ix+1) % args.break_interval == 0:
            break

    mu_loss /= min(ix+1, len(train_loader))
    mu_recon /= min(ix+1, len(train_loader))
    mu_kl_div /= min(ix + 1, len(train_loader))

    if epoch % args.log_interval == 0:
        print('Train Epoch: {} Loss: {} Recon: {} KL: {}'.format(
            epoch, mu_loss, mu_recon, mu_kl_div))

    if epoch % args.checkpoint_interval == 0:
        args = add_params_to_args(
            args, mu_loss, mu_recon, mu_kl_div, 0, 'train', epoch)

        df = utils.write_row_to_df(args, df)
        print(df.tail())

        #save_model(model, epoch, args)

    return df


def add_params_to_args(args, mu_loss, mu_recon, mu_kl_div, mu_marginal_lik, mode, epoch):
    args.mu_loss = mu_loss
    args.recon_loss = mu_recon
    args.kl_loss = mu_kl_div
    args.mode = mode
    args.epoch = epoch
    args.marginal_likelihood = mu_marginal_lik

    return args


def test(model, test_loader, epoch, args, df=None):
    model.eval()

    mu_loss = 0
    mu_recon = 0
    mu_kl_div = 0

    mu_marginal_lik = 0

    with torch.no_grad():
        for ix, (xsample, label) in enumerate(test_loader):

            if args.dataset in ['mnist']:
                # Flatten
                xsample = xsample.view(xsample.size(0), -1)

            recon, kl_div = model.loss_fn(xsample)

            loss = recon + args.beta * kl_div
            mu_loss += loss.item()
            mu_recon += recon.item()
            mu_kl_div += kl_div.item()

            marginal_lik = helpers.marginal_likelihood(
                model, xsample, args.marginal_samples)
            mu_marginal_lik += marginal_lik.item()

            if args.break_early == 1 and (ix+1) % args.break_interval == 0:
                break

    mu_loss /= min(ix+1, len(test_loader))
    mu_recon /= min(ix+1, len(test_loader))
    mu_kl_div /= min(ix+1, len(test_loader))
    mu_marginal_lik /= min((ix+1) * len(xsample), len(test_loader.dataset))

    model_name = model_name_from_config(args)
    file_name = os.path.join(
        IMAGES_DIR, f'test_img_{epoch}_{model_name}' + '.png')
    utils.plot_and_save_fig(model, args.c, test_loader,
                            file_name, typ=args.model, args=args)

    print('Test set: average loss: {:.4f}'.format(mu_loss))
    print('Test set: marginal likelihood: {:.4f}'.format(mu_marginal_lik))
    args = add_params_to_args(
        args, mu_loss, mu_recon, mu_kl_div, mu_marginal_lik, 'test', epoch)

    df = utils.write_row_to_df(args, df)

    return df


def save_model(model, epoch, args):
    model_name = f'chkpt-epoch-{epoch}-' + \
        model_name_from_config(args)
    checkpoint_path = os.path.join(ROOT_DIR, 'checkpoints')
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    torch.save(model, os.path.join(
        checkpoint_path, model_name + '.pt'))


def model_name_from_config(args):
    return f"{args.model_uuid}-{args.distribution}-{args.dataset}-{args.latent_dim}-{args.loss}-prior-{args.prior_sigma}-c-{args.c}-seed-{args.seed}"


def set_up_dirs():
    if not os.path.exists(ROOT_DIR):
        os.mkdir(ROOT_DIR)
    if not os.path.exists(IMAGES_DIR):
        os.mkdir(IMAGES_DIR)
    if not os.path.exists(MODELS_DIR):
        os.mkdir(MODELS_DIR)
    if not os.path.exists(CSV_DIR):
        os.mkdir(CSV_DIR)


def main(config, df=None):
    torch.manual_seed(config.seed)
    model_name = model_name_from_config(config)

    set_up_dirs()

    if config.dataset in ['toy']:
        input_dim = config.toy_dim
        latent_dim = config.latent_dim
        loader = BranchingLoaders(
            dim=input_dim, sigma=1, sigma_x=1,
            num_x=5,
            max_depth=7,
            batch_size=config.batch_size,
            test_batch=config.test_batch,
            seed=config.seed)

        train_loader, test_loader = loader.get_data_loaders()

    if config.dataset in ['mnist']:
        if config.loss in ['mse']:
            train_loader = DataLoader(MNIST('data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
                batch_size=config.batch_size, shuffle=True)

            test_loader = DataLoader(MNIST('data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
                batch_size=config.test_batch, shuffle=True)

        elif config.loss in ['bce']:
            EPS = 1e-6
            train_loader = DataLoader(MNIST('data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda p: p.clamp(EPS, 1-EPS))
            ])),
                batch_size=config.batch_size, shuffle=True)

            test_loader = DataLoader(MNIST('data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda p: p.clamp(EPS, 1-EPS))
            ])),
                batch_size=config.test_batch, shuffle=True)

        input_dim = 28*28
        latent_dim = config.latent_dim

    if config.model in ['hyperbolic']:
        model = HyperbolicVAE(input_dim, latent_dim=latent_dim, h=config.hidden_dim, gyroplane_dim=config.hidden_dim,
                              c=config.c, dist=config.distribution, prior_sigma=config.prior_sigma)

    if config.model in ['vanilla']:
        model = VanillaVAE(input_dim, latent_dim, h=config.hidden_dim,
                           prior_sigma=config.prior_sigma)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    import gc
    gc.collect()
    for epoch in range(1, config.epochs + 1):
        df = train(model, optimizer, train_loader, epoch, config, df)
        if epoch % config.test_interval == 0:
            try:
                df = test(model, test_loader, epoch, config, df)
            except Exception as e:
                print("Error testing model")
                print(e)
                print(traceback.format_exc())

    # Final score
    try:
        df = test(model, test_loader, epoch, config, df)
    except:
        print("error in final test")

    file_name = os.path.join(IMAGES_DIR, model_name + '.png')
    utils.plot_and_save_fig(model, config.c, test_loader,
                            file_name, typ=config.model, args=config)

    if config.save_model == 1:
        torch.save(model, os.path.join(
            MODELS_DIR, model_name + ".pt"))

    if df is not None:
        df.to_csv(os.path.join(CSV_DIR, "final_" + model_name + '.csv'))
    return df


if __name__ == '__main__':
    cfg = vars(config)
    df = pd.DataFrame(columns=cfg.keys())
    main(config, df)
