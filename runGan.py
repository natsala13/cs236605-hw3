# Lets try an experiment

from hw3.answers import part2_vae_hyperparams
from hw3.experiments import run_experiment_GAN
# from hw3.autoencoder import AutoEncoderError 
import numpy as np
     
Init_Name = 'Gan_First_Run'
OUTDIR = './results/Res4/'

AllResults = {}

bs = 16
h_dim = 256
z_dim = 128
Epochs = 100
lr = 0.0005

data_label = 1
label_noise = 0.3



def oneExp(gen_lr,des_lr,generator_optim,dsc_optim):
    try:
        name = Init_Name + 'DSC_' + dsc_optim + 'GEN_' + generator_optim + 'genlr_' + str(gen_lr) + 'deslr' + str(des_lr)
        res = run_experiment_GAN(name, out_dir=OUTDIR, seed=42,
                                # Training params
                                bs_train=8, bs_test=None, batches=100, epochs=100,
                                early_stopping=10, checkpoints=None,print_every=100,
                                # Model params
                                h_dim=256, z_dim=128, gen_lr=gen_lr, des_lr=des_lr, 
                                 generator_optim=generator_optim,
                                 dsc_optim=dsc_optim, data_label=data_label, label_noise=label_noise)
        AllResults[name] = res
    except OSError as e:
        AllResults[name] = 'Failed... ' + str(e)
    
    
    
            


for gen_lr in [0.0008,0.0005,0.0001]:
    for des_lr in [0.0008,0.0005,0.0001]:
        for generator_optim in ['SGD', 'ADAM']:
            for dsc_optim in ['SGD', 'ADAM']:
                oneExp(gen_lr,des_lr,generator_optim,dsc_optim)
            
            
            
np.save(OUTDIR + 'RunFinal', AllResults)

