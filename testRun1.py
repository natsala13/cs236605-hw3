# Lets try an experiment

from hw3.answers import part2_vae_hyperparams
from hw3.experiments import run_experiment


# Hyperparams
hp = part2_vae_hyperparams()
batch_size = hp['batch_size']
h_dim = hp['h_dim']
z_dim = hp['z_dim']
x_sigma2 = hp['x_sigma2']
learn_rate = hp['learn_rate']
betas = hp['betas']

     
Init_Name = 'MEGA_RUN'

for lr in [0.01,0.005,0.001,0.0005]:
    for s in [0.5,0.9]:
        name = Init_Name + 'lr_' + str(lr) + 's_' + str(s)
        run_experiment(name, out_dir='./results', seed=42,
                        # Training params
                        bs_train=32, bs_test=None, batches=100, epochs=30,
                        early_stopping=3, checkpoints=None, lr=lr,
                        # Model params
                        h_dim=256, z_dim=5, x_sigma2=0.9)