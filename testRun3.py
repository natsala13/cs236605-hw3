# Lets try an experiment

from hw3.answers import part2_vae_hyperparams
from hw3.experiments import run_experiment
from hw3.autoencoder import AutoEncoderError 
import numpy as np
     
Init_Name = 'MEGA_RUN'
OUTDIR = './results/Res1/'

AllResults = {}

bs = 16
Epochs = 40
# s = 0.95
lr = 0.0005
h = 256
z = 128


def oneExp(bs,s):
    try:
        name = 's_' + str(s) + 'bs_' + str(bs)
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



for s in [0.8,0.85,0.9,1]:
    for bs in [16,32]:
        oneExp(bs,s)
            

            
            
np.save(OUTDIR + 'RunFinal', AllResults)

