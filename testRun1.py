# Lets try an experiment

from hw3.answers import part2_vae_hyperparams
from hw3.experiments import run_experiment
from hw3.autoencoder import AutoEncoderError 
import numpy as np
     
Init_Name = 'MEGA_RUN'
OUTDIR = './results/Res1/'

AllResults = {}

bs = 32
h_dim = 256
z_dim = 10
Epochs = 100




for lr in [0.01,0.005,0.001,0.0005]:
    for s in [0.5,0.9]:
        try:
            name = 'lr_' + str(lr) + 's_' + str(s)
            res = run_experiment(Init_Name + name, out_dir=OUTDIR, seed=42,
                            # Training params
                            bs_train=32, bs_test=None, batches=100, epochs=Epochs,
                            early_stopping=10, checkpoints=None, lr=lr,
                            # Model params
                            h_dim=h_dim, z_dim=z_dim, x_sigma2=s)
            AllResults[name] = res
        except AutoEncoderError as e:
            AllResults[name] = 'Failed... ' + str(e)
        except OSError as e:
            AllResults[name] = 'Failed... ' + str(e)
            

            
            
np.save(OUTDIR + 'Run1Final', AllResults)
