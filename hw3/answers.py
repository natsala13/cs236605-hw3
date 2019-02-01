r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_generation_params():
    start_seq = ""
    temperature = .0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    batch_size = 32
    h_dim = 256
    z_dim = 64
    x_sigma2 = 0.8
    learn_rate = 0.0005
    betas = (0.5,0.5)
    
    
    hypers = dict(
        batch_size=batch_size,
        h_dim=h_dim, z_dim=z_dim, x_sigma2=x_sigma2,
        learn_rate=learn_rate, betas=betas,
    )
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


$\sigma$ acts as a regularisation strength. since our loss is composed of a Data-loss and a a distrobution-loss (KLdiv), the parameter $\sigma$ defines the strength of the Data-loss. when sigma is high, the weight of the Data-loss is smaller so the regularisation is stronger. and when sigma is low the weight of the Data-loss is much bigger so the model try to fit more to the data itself.

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 32
    hypers['z_dim'] = 128
    hypers['data_label'] = 1
    hypers['label_noise'] = 0.3
    
    hypers['discriminator_optimizer']['type'] = 'SGD'
    hypers['discriminator_optimizer']['lr'] = 0.0005
    
    hypers['generator_optimizer']['type'] = 'SGD'
    hypers['generator_optimizer']['lr'] = 0.0008
    
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**


While training we train the descriminator once and then the generator for every batch. Thus we do not want the gnerator to train while we train the Discriminator using the Discriminator loss etc, so we don't pass backward on the sampler every time we sample for training the Discriminator.

"""

part3_q2 = r"""
**Your answer:**

1) When training a GAN we should not stop training based on the fact that the Generator loss is low since the if the Discriminator loss is high, it means the descriminator is not credible. Since the low loss of the Generator only means that the descriminator cant identify real and fakes images and the discriminator is not well traind we cannot trust the Generator loss and so should not stop the training.


2) Since the Generator loss is decreassing it means the Generator is being better so it produces images closer to real ones, so the Discriminator would have more trouble distinguishing between them and it would probably make decision rules that could affect the classifications of real ones, so the Discriminator would possibly improve the classification on generated images  and decrease the classification of real images (since the generated images are closer to real ones) and so the loss would remain stable more or less.


"""

part3_q3 = r"""
**Your answer:**

Compare the results you got when generating images with the VAE to the GAN results. What's the main difference and what's causing it?




"""

# ==============


