# Lets try an experiment

from hw3.answers import part2_vae_hyperparams
from hw3.experiments import run_experiment_VAE
from hw3.autoencoder import AutoEncoderError 
import numpy as np
     
Init_Name = 'MEGA_RUN'
OUTDIR = './results/Res6/'

AllResults = {}

bs = 32
h_dim = 256
z_dim = 64
Epochs = 100
lr = 0.0005
s = 0.8
betas = (0.5,0.5)



try:           
    name = 'without_Pooling'
    res = run_experiment_VAE(Init_Name + name, out_dir=OUTDIR, seed=42,
                            # Training params
                            bs_train=8, bs_test=None, batches=100, epochs=Epochs,
                            early_stopping=10, checkpoints=None, lr=lr,betas=betas,print_every=10,
                            # Model params
                            h_dim=h_dim, z_dim=z_dim, x_sigma2=s)
    AllResults[name] = res
except AutoEncoderError as e:
    AllResults[name] = 'Failed... ' + str(e)
except OSError as e:
    AllResults[name] = 'Failed... ' + str(e)
    
    
    
            
np.save(OUTDIR + 'RunFinal', AllResults)
