# Lets try an experiment

from hw3.answers import part2_vae_hyperparams
from hw3.experiments import run_experiment_GAN
# from hw3.autoencoder import AutoEncoderError 
import numpy as np
     
Init_Name = 'MEGA_RUN'
OUTDIR = './results/Res3/'

AllResults = {}

bs = 16
h_dim = 256
z_dim = 128
Epochs = 100
lr = 0.0005




try:
    name = 'Gan_First_Run'
    res = run_experiment_GAN(name, out_dir=OUTDIR, seed=42,
                            # Training params
                            bs_train=8, bs_test=None, batches=100, epochs=100,
                            early_stopping=10, checkpoints=None,print_every=5,
                            # Model params
                            h_dim=256, z_dim=128)
    AllResults[name] = res
except OSError as e:
    AllResults[name] = 'Failed... ' + str(e)
    
    
    
            
np.save(OUTDIR + 'RunFinal', AllResults)
