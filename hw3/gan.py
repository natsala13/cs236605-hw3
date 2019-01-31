from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from .autoencoder import EncoderCNN, DecoderCNN


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        # To extract image features you can use the EncoderCNN from the VAE
        # section or implement something new.
        # You can then use either an affine layer or another conv layer to
        # flatten the features.
        # ====== YOUR CODE: ======
        in_channels = in_size[0]
        out_channels = 256
        
        out_spatial = 8 # The encoder returns (out_channels,8,8)

        
        self.feature_extractor = EncoderCNN(in_channels,out_channels)
    
        # Now lets create the classifier part
        # After convolution we get (ou_channels,8,8)
        Cin = out_channels * out_spatial * out_spatial
        hidden_dims = [2048,256,1]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        modules = []        
        for Cout in hidden_dims:
            modules += [nn.Linear(Cin,Cout).to(device),nn.LeakyReLU()]
            Cin = Cout
        
        self.classifier = nn.Sequential(*modules)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (aka logits, not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        # No need to apply sigmoid to obtain probability - we'll combine it
        # with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        x = self.feature_extractor(x)
        
        N = x.shape[0]
        x = x.view(N,-1)
        
        y = self.classifier(x)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        # To combine image features you can use the DecoderCNN from the VAE
        # section or implement something new.
        # You can assume a fixed image size.
        # ====== YOUR CODE: ======
        Cin = z_dim
        hidden_dims = [256,2048,16384]
        self.out_spatial = 8 # The encoder returns (out_channels,8,8)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        modules = []        
        for Cout in hidden_dims:
            modules += [nn.Linear(Cin,Cout).to(device),nn.LeakyReLU()]
            Cin = Cout
        
        self.transform = nn.Sequential(*modules)
        
#         print('Cin - ', Cin / (self.out_spatial * self.out_spatial))
        Cin = Cin // (self.out_spatial * self.out_spatial)
        
        
        
        # We will use the Model from Part 2
        self.generator = DecoderCNN(Cin,out_channels)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should have
        gradients or not.
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        # Generate n latent space samples and return their reconstructions.
        # Don't use a loop.
        # ====== YOUR CODE: ======
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            

        
        with torch.set_grad_enabled(with_grad):
#             z = torch.randn(n, self.z_dim, device=device)
#             samples = self(z)
            o = torch.zeros(n,self.z_dim)
            z = torch.normal(o,1).to(device)

            samples = self.forward(z)
        
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        # Don't forget to make sure the output instances have the same scale
        # as the original (real) images.
        # ====== YOUR CODE: ======
        z = self.transform(z)
        
        N = z.shape[0]
        z = z.view(N,-1,self.out_spatial,self.out_spatial)
        
        x = self.generator(z)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO: Implement the discriminator loss.
    # See torch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    noise1 = label_noise * torch.rand_like(y_data) - (label_noise/2)
    noise2 = label_noise * torch.rand_like(y_data) - (label_noise/2)
    
    loss_data = F.binary_cross_entropy_with_logits(y_data, data_label + noise1)
    loss_generated = F.binary_cross_entropy_with_logits(y_generated, 1 - data_label + noise2)
    
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    # TODO: Implement the Generator loss.
    # Think about what you need to compare the input to, in order to
    # formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    label = torch.ones_like(y_generated) * data_label
    loss = F.binary_cross_entropy_with_logits(y_generated, label)
    # ========================
    return loss


def train_batch(dsc_model: Discriminator, gen_model: Generator,
                dsc_loss_fn: Callable, gen_loss_fn: Callable,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_data: DataLoader):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    # 1. Show the discriminator real and generated data
    # 2. Calculate discriminator loss
    # 3. Update discriminator parameters
    # ====== YOUR CODE: ======
    N = x_data.shape[0]
    
    # Set params for training the Discriminator
    dsc_model.train(mode=True)
    gen_model.train(mode=False)    
    dsc_optimizer.zero_grad()
    
    # Generate new data using Generator - ! We Do Not Train The Generator ! 
    x_fake = gen_model.sample(N, with_grad=False)
    
    # Calculate scores for both real and fake images
    real_prob = dsc_model(x_data)
    fake_prob = dsc_model(x_fake)
    
    # Calculate loss
    dsc_loss = dsc_loss_fn(real_prob, fake_prob)
    
    # Train
    dsc_loss.backward()
    dsc_optimizer.step()
    
    # ========================

    # TODO: Generator update
    # 1. Show the discriminator generated data
    # 2. Calculate generator loss
    # 3. Update generator parameters
    # ====== YOUR CODE: ======
    N = x_data.shape[0]
    
    # Set params for training the Discriminator
    dsc_model.train(mode=False)
    gen_model.train(mode=True)    
    gen_optimizer.zero_grad()
    
    # Generate new data using Generator
    x_fake = gen_model.sample(N, with_grad=True)
    
    # Calculate scores for both real and fake images
    real_prob = dsc_model(x_data)
    fake_prob = dsc_model(x_fake)
    
    # Calculate loss
    gen_loss = gen_loss_fn(fake_prob)
    
    # Train
    gen_loss.backward()
    gen_optimizer.step()
    

    # Finally set both model's training off
    dsc_model.train(mode=False)
    gen_model.train(mode=False)   
    # ========================

    return dsc_loss.item(), gen_loss.item()

