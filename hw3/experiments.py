import argparse
import itertools
import os
import random
import sys
import json
import pathlib
import tqdm

import torch
import torchvision

import cs236605.download
import cs236605.plot as plot
import matplotlib.pyplot as plt

from cs236605.train_results import FitResult, GANResult

import torchvision.transforms as T
from torchvision.datasets import ImageFolder

import hw3.autoencoder as autoencoder
from hw3.autoencoder import vae_loss

import hw3.gan as gan
# from hw3.gan import *


import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from hw3.training import VAETrainer
from hw3.answers import part2_vae_hyperparams


import IPython.display

import numpy as np

# from . import models
# from . import training

DATA_DIR = os.path.join(os.getenv('HOME'), '.pytorch-datasets')
DATA_URL = 'http://vis-www.cs.umass.edu/lfw/lfw-bush.zip'

def run_experiment_VAE(run_name, out_dir='./results', seed=42,
                   # Training params
                   bs_train=128, bs_test=None, batches=100, epochs=100,
                   early_stopping=3, checkpoints=None, lr=1e-3,RTplot=False,
                   print_every=100,
                   # Model params
                   h_dim=256, z_dim=5, x_sigma2=0.9,betas=(0.9,0.999),
                   **kw):
    """
        Execute a single run of experiment 1 with a single configuration.
        :param run_name: The name of the run and output file to create.
        :param out_dir: Where to write the output to.
        """

    torch.manual_seed(seed)
    if not bs_test:
        bs_test = max([bs_train // 4, 1])
    cfg = locals()

    
    DATA_DIR = pathlib.Path.home().joinpath('.pytorch-datasets')
    _, dataset_dir = cs236605.download.download_data(out_path=DATA_DIR, url=DATA_URL, extract=True, force=False)
    im_size = 64
    tf = T.Compose([
        # Resize to constant spatial dimensions
        T.Resize((im_size, im_size)),
        # PIL.Image -> torch.Tensor
        T.ToTensor(),
        # Dynamic range [0,1] -> [-1, 1]
        T.Normalize(mean=(.5,.5,.5), std=(.5,.5,.5)),
    ])
    
    ds_gwb = ImageFolder(os.path.dirname(dataset_dir), tf)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    
    # TODO: Train
    # - Create model, loss, optimizer and trainer based on the parameters.
    #   Use the model you've implemented previously, cross entropy loss and
    #   any optimizer that you wish.
    # - Run training and save the FitResults in the fit_res variable.
    # - The fit results and all the experiment parameters will then be saved
    #  for you automatically.
    fit_res = None
    # ====== YOUR CODE: ======
    
    
    # Data
    split_lengths = [int(len(ds_gwb)*0.9), int(len(ds_gwb)*0.1)]
    ds_train, ds_test = random_split(ds_gwb, split_lengths)
    dl_train = DataLoader(ds_train, bs_train, shuffle=True)
    dl_test  = DataLoader(ds_test,  bs_train, shuffle=True)
    im_size = ds_train[0][0].shape

    # Model
    encoder = autoencoder.EncoderCNN(in_channels=im_size[0], out_channels=h_dim)
    decoder = autoencoder.DecoderCNN(in_channels=h_dim, out_channels=im_size[0])
    vae = autoencoder.VAE(encoder, decoder, im_size, z_dim)
    vae_dp = DataParallel(vae).to(device)
    print(vae)

    # Optimizer
    optimizer = optim.Adam(vae.parameters(), lr=lr, betas=betas)
    
    
    # Loss
    def loss_fn(x, xr, z_mu, z_log_sigma2):
        return autoencoder.vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2)
    
    def post_epoch_fn(epoch, train_result, test_result, verbose):
        # Plot some samples if this is a verbose epoch
        if verbose:
            samples = vae.sample(n=5)
            fig, _ = plot.tensors_as_images(samples, figsize=(6,2))
            if RTplot:
                IPython.display.display(fig)
            else:
                name = run_name + '_Ep_' + str(epoch)
                fig.savefig(out_dir + name + '.png')
            plt.close(fig)

    # Trainer
    trainer = VAETrainer(vae_dp, loss_fn, optimizer, device)
    checkpoint_file = 'checkpoints/vae'
    checkpoint_file_final = f'{checkpoint_file}_final'
    if os.path.isfile(f'{checkpoint_file}.pt'):
        os.remove(f'{checkpoint_file}.pt')
    
    


    fit_res = trainer.fit(dl_train, dl_test,
                  num_epochs=epochs, early_stopping=20, print_every=print_every,
                  checkpoints=checkpoint_file,
                  post_epoch_fn=post_epoch_fn)
    
    last_train_loss = fit_res.train_loss[-1]
    last_test_loss = fit_res.test_loss[-1]
    # ========================
    
    save_experiment(run_name, out_dir, cfg, fit_res)
    
    
    return {'train': last_train_loss, 'test': last_test_loss}
    


    
def run_experiment_GAN(run_name, out_dir='./results', seed=42,
                   # Training params
                   bs_train=128, bs_test=None, batches=100, epochs=100,
                   early_stopping=3, checkpoints=None, gen_lr=1e-3, des_lr=1e-3, RTplot=False,
                   print_every=100, generator_optim='SGD', dsc_optim='SGD',
                    data_label = 1, label_noise=0.3,
                   # Model params
                   h_dim=256, z_dim=5,
                   **kw):
#         """
#         Execute a single run of experiment 1 with a single configuration.
#         :param run_name: The name of the run and output file to create.
#         :param out_dir: Where to write the output to.
#         """

    torch.manual_seed(seed)
    cfg = locals()

    ############### Download DataSet #################
    DATA_DIR = pathlib.Path.home().joinpath('.pytorch-datasets')
    _, dataset_dir = cs236605.download.download_data(out_path=DATA_DIR, url=DATA_URL, extract=True, force=False)
    im_size = 64
    tf = T.Compose([
        # Resize to constant spatial dimensions
        T.Resize((im_size, im_size)),
        # PIL.Image -> torch.Tensor
        T.ToTensor(),
        # Dynamic range [0,1] -> [-1, 1]
        T.Normalize(mean=(.5,.5,.5), std=(.5,.5,.5)),
    ])
    ds_gwb = ImageFolder(os.path.dirname(dataset_dir), tf)
    ##################################################
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    
    # TODO: Train
    # - Create model, loss, optimizer and trainer based on the parameters.
    #   Use the model you've implemented previously, cross entropy loss and
    #   any optimizer that you wish.
    # - Run training and save the FitResults in the fit_res variable.
    # - The fit results and all the experiment parameters will then be saved
    #  for you automatically.
    fit_res = None
    # =========== Prepare For Training: ============

    # Hyperparams

    # Data
    dl_train = DataLoader(ds_gwb, bs_train, shuffle=True)
    im_size = ds_gwb[0][0].shape

    # Model
    dsc = gan.Discriminator(im_size).to(device)
    gen = gan.Generator(z_dim, featuremap_size=4).to(device)

    # Optimizer
    def create_optimizer(model_params, opt_params):
        opt_params = opt_params.copy()
        optimizer_type = opt_params['type']
        opt_params.pop('type')
        return optim.__dict__[optimizer_type](model_params, **opt_params)
    
    generator_optimizer=dict(type=generator_optim,lr=gen_lr)
    discriminator_optimizer=dict(type=dsc_optim,lr=des_lr)
    
    dsc_optimizer = create_optimizer(dsc.parameters(), discriminator_optimizer)
    gen_optimizer = create_optimizer(gen.parameters(),generator_optimizer)

    # Loss
    def dsc_loss_fn(y_data, y_generated):
        return gan.discriminator_loss_fn(y_data, y_generated, data_label,label_noise)

    def gen_loss_fn(y_generated):
        return gan.generator_loss_fn(y_generated,data_label)

    # Training
    checkpoint_file = 'checkpoints/gan'
    checkpoint_file_final = f'{checkpoint_file}_final'
    if os.path.isfile(f'{checkpoint_file}.pt'):
        os.remove(f'{checkpoint_file}.pt')
        
        
    def post_epoch_fn(epoch):
        # Plot some samples if this is a verbose epoch
        samples = gen.sample(n=5)
        fig, _ = plot.tensors_as_images(samples.cpu(), figsize=(6,2))
        if RTplot:
            IPython.display.display(fig)
        else:
            name = run_name + '_Ep_' + str(epoch)
            fig.savefig(out_dir + name + '.png')
        plt.close(fig)
    
    # ================================================
    
    # ============== Actual Training =================    

    if os.path.isfile(f'{checkpoint_file_final}.pt'):
        print(f'*** Loading final checkpoint file {checkpoint_file_final} instead of training')
        num_epochs = 0
        gen = torch.load(f'{checkpoint_file_final}.pt', map_location=device)
        checkpoint_file = checkpoint_file_final
    
    fit_gen_loss = []
    fit_des_loss = []
    
    for epoch_idx in range(epochs):
        # We'll accumulate batch losses and show an average once per epoch.
        dsc_losses = []
        gen_losses = []
        print(f'--- EPOCH {epoch_idx+1}/{epochs} ---')

        with tqdm.tqdm(total=len(dl_train.batch_sampler), file=sys.stdout) as pbar:
            for batch_idx, (x_data, _) in enumerate(dl_train):
                x_data = x_data.to(device)
                dsc_loss, gen_loss = gan.train_batch(
                    dsc, gen,
                    dsc_loss_fn, gen_loss_fn,
                    dsc_optimizer, gen_optimizer,
                    x_data)
                dsc_losses.append(dsc_loss)
                gen_losses.append(gen_loss)
                pbar.update()

        dsc_avg_loss, gen_avg_loss = np.mean(dsc_losses), np.mean(gen_losses)
        print(f'Discriminator loss: {dsc_avg_loss}')
        print(f'Generator loss:     {gen_avg_loss}')
        
        # save final epochs' loss
        fit_gen_loss.append(gen_avg_loss)
        fit_des_loss.append(dsc_avg_loss)

#         samples = gen.sample(5, with_grad=False)
#         fig, _ = plot.tensors_as_images(samples.cpu(), figsize=(6,2))
#         IPython.display.display(fig)
#         plt.close(fig)
        if epoch_idx % print_every == 0 or epoch_idx == epochs - 2:
            post_epoch_fn(epoch_idx)
    
    fit_res = GANResult(epochs,fit_gen_loss,fit_des_loss)
    
    last_dis_loss = fit_des_loss[-1]
    last_gen_loss = fit_gen_loss[-1]
    # ================================================
    
    
    save_experiment(run_name, out_dir, cfg, fit_res)
    
    
    return {'Gen': last_gen_loss, 'Dis': last_dis_loss}
    

    

    


    
RESULT_DIR = 'results/'





    
    
    
def save_experiment(run_name, out_dir, config, fit_res):
    output = dict(
                  config=config,
                  results=fit_res._asdict()
                  )
    output_filename = f'{os.path.join(out_dir, run_name)}.json'
    os.makedirs(out_dir, exist_ok=True)
    with open(output_filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'*** Output file {output_filename} written')


def load_experiment(filename, resultClass=FitResult):
    with open(filename, 'r') as f:
        output = json.load(f)


    fit_res = resultClass(**output['results'])

    return None, fit_res



def parse_cli():
    p = argparse.ArgumentParser(description='CS236605 HW2 Experiments')
    sp = p.add_subparsers(help='Sub-commands')

    # Experiment config
    sp_exp = sp.add_parser('run-exp', help='Run experiment with a single '
                                           'configuration')
    sp_exp.set_defaults(subcmd_fn=run_experiment)
    sp_exp.add_argument('--run-name', '-n', type=str,
                        help='Name of run and output file', required=True)
    sp_exp.add_argument('--out-dir', '-o', type=str, help='Output folder',
                        default='./results', required=False)
    sp_exp.add_argument('--seed', '-s', type=int, help='Random seed',
                        default=None, required=False)

    # # Training
    sp_exp.add_argument('--bs-train', type=int, help='Train batch size',
                        default=128, metavar='BATCH_SIZE')
    sp_exp.add_argument('--bs-test', type=int, help='Test batch size',
                        metavar='BATCH_SIZE')
    sp_exp.add_argument('--batches', type=int,
                        help='Number of batches per epoch', default=100)
    sp_exp.add_argument('--epochs', type=int,
                        help='Maximal number of epochs', default=100)
    sp_exp.add_argument('--early-stopping', type=int,
                        help='Stop after this many epochs without '
                             'improvement', default=3)
    sp_exp.add_argument('--checkpoints', type=int,
                        help='Save model checkpoints to this file when test '
                             'accuracy improves', default=None)
    sp_exp.add_argument('--lr', type=float,
                        help='Learning rate', default=1e-3)
    sp_exp.add_argument('--reg', type=int,
                        help='L2 regularization', default=1e-3)

    # # Model
    sp_exp.add_argument('--filters-per-layer', '-K', type=int, nargs='+',
                        help='Number of filters per conv layer in a block',
                        metavar='K', required=True)
    sp_exp.add_argument('--layers-per-block', '-L', type=int, metavar='L',
                        help='Number of layers in each block', required=True)
    sp_exp.add_argument('--pool-every', '-P', type=int, metavar='P',
                        help='Pool after this number of conv layers',
                        required=True)
    sp_exp.add_argument('--hidden-dims', '-H', type=int, nargs='+',
                        help='Output size of hidden linear layers',
                        metavar='H', required=True)
    sp_exp.add_argument('--ycn', action='store_true', default=False,
                        help='Whether to use your custom network')

    parsed = p.parse_args()

    if 'subcmd_fn' not in parsed:
        p.print_help()
        sys.exit()
    return parsed
if __name__ == '__main__':
    parsed_args = parse_cli()
    subcmd_fn = parsed_args.subcmd_fn
    del parsed_args.subcmd_fn
    print(f'*** Starting {subcmd_fn.__name__} with config:\n{parsed_args}')
    subcmd_fn(**vars(parsed_args))
