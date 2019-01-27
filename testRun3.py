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


for bs in [16,32,64]:
    for lr inm [0.01,0.005,0.001,0.0005]:
        for h in [128,256,512]:
            for z in [2,5,10,20,50]:
                for s in [0.3,0.5,0.7,0.9]:
                    name = 'bs_' + str(bs) + 'lr_' + str(lr) + 'h_' + str(h) + 'z_' + str(z) + 's_' + str(s)
                    run_experiment(name, out_dir='./results', seed=42,
                                    # Training params
                                   bs_train=bs, bs_test=None, batches=100, epochs=30,
                                   early_stopping=3, checkpoints=None, lr=lr,
                                   # Model params
                                   h_dim=h, z_dim=z, x_sigma2=s)


