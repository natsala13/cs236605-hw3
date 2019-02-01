# Lets try an experiment

from hw3.answers import part2_vae_hyperparams
from hw3.experiments import run_experiment
from hw3.autoencoder import AutoEncoderError 
import numpy as np
     
Init_Name = 'FINAL_ARCH'
OUTDIR = './results/Res2/'

AllResults = {}

# bs = 16
Epochs = 60
s = 0.9
lr = 0.0005
h = 256
z = 128


def oneExp(lr,bs):
    try:
        name = 'bs_' + str(bs) + 'lr_' + str(lr)
        res = run_experiment(Init_Name + name, out_dir=OUTDIR, seed=42,
                             # Training params
                             bs_train=bs, bs_test=None, batches=100, epochs=Epochs,
                             early_stopping=10, checkpoints=None, lr=lr,
                             # Model params
                             h_dim=h, z_dim=z, x_sigma2=s)
        
        AllResults[name] = res
    except AutoEncoderError as e:
        AllResults[name] = 'Failed... ' + str(e)
    except OSError as e:
        AllResults[name] = 'Failed... ' + str(e)


for bs in [16,32,64]:
    for lr in [0.0008,0.0005,0.0001]:
        oneExp(lr,bs)
            
            
            
np.save(OUTDIR + 'RunFinal', AllResults)


