import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO: Implement a CNN. Save the layers in the modules list.
        # The input shape is an image batch: (N, in_channels, H_in, W_in).
        # The output shape should be (N, out_channels, H_out, W_out).
        # You can assume H_in, W_in >= 64.
        # Architecture is up to you, but you should use at least 3 Conv layers.
        # You can use any Conv layer parameters, use pooling or only strides,
        # use any activation functions, use BN or Dropout, etc.
        # ====== YOUR CODE: ======
        
        modules = []
        Cin = in_channels
        convs = [64,128] + [out_channels]
        for Cout in convs:
#             modules += [nn.Conv2d(Cin,Cout,5,padding=2),nn.MaxPool2d(2),nn.BatchNorm2d(Cout),nn.ReLU()]
            modules += [nn.Conv2d(Cin,Cout,5,stride=2,padding=2),nn.BatchNorm2d(Cout),nn.ReLU()]
            Cin = Cout

        
#         modules += [nn.MaxPool2d(4),nn.Conv2d(Cin,out_channels,5,padding=2)]
#         modules += [nn.Conv2d(Cin,out_channels,5,padding=2)]
            
            
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO: Implement the "mirror" CNN of the encoder.
        # For example, instead of Conv layers use transposed convolutions,
        # instead of pooling do unpooling (if relevant) and so on.
        # You should have the same number of layers as in the Encoder,
        # and they should produce the same volumes, just in reverse order.
        # Output should be a batch of images, with same dimensions as the
        # inputs to the Encoder were.
        # ====== YOUR CODE: ======
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        Cin = in_channels
        convs = [32,128,256]
        modules = []
        
        

        for Cout in reversed(convs):
            modules += [nn.ConvTranspose2d(Cin,Cout,5,stride=2,padding=2,output_padding=1)]
            modules += [nn.BatchNorm2d(Cout).to(device)]
            modules += [nn.ReLU()]  
            Cin = Cout
            
        modules += [nn.ConvTranspose2d(Cin,out_channels,5,padding=2)]
    
        
        
        self.conv1 = nn.ConvTranspose2d(in_channels,128,5,stride=2,padding=2,output_padding=1)
        self.bn1 = nn.BatchNorm2d(128).to(device)
        self.rl = nn.ReLU()
        
        # ========================
        self.modules = modules
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        print('h device - ', h.device)
#         print('cnn device - ' , self.cnn.device)
    
    
        h = self.conv1(h)
        print(h)
        h = self.bn1(h)
        print(type(h))
        print('h shape - ', h.shape)
        print(h[0].cpu())
        
        h = h.cpu()
        
#         try:
#             h = self.rl(h_cpu)
#         except Exception as e:
#             print('########################### First time error ####################################')
#             print(str(e))
#             print('##########################################################################')
        
#         h_gpu = h_cpu.cuda()
        
#         try:
#             h = self.rl(h_gpu)
#         except Exception as e:
#             print('######################Second time error#################################')
#             print(str(e))
#             print('##########################################################################')
        
        
        h = self.rl(h)
    
    
    
#         for m in self.modules:
#             print(m)
#             try:
#                 print('m device - ', m.device)
#             except:
#                 print('################ m has no device ##############')
#             h = m(h)
    
    
#         x = self.cnn(h)
        return torch.tanh(h)


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder the extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from it's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)

        # TODO: Add parameters needed for encode() and decode().
        # ====== YOUR CODE: ======
        self.spatial = 8
        
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        D = self.features_shape[0] * self.spatial * self.spatial
        
#         m_w = torch.zeros(D,z_dim)
#         m_b = torch.zeros(1,z_dim)
        
#         s = 1 / D
        
#         self.Whu = torch.normal(m_w,s).to(device)
#         self.Bhu = torch.normal(m_b,1).to(device)
#         self.Whs = torch.normal(m_w,s).to(device)
#         self.Bhs = torch.normal(m_b,1).to(device)
        
        
        
#         m_w = torch.zeros(z_dim,D)
#         m_b = torch.zeros(1,D)
        
#         self.Dec_T = torch.normal(m_w,1).to(device)
#         self.Dec_b = torch.normal(m_b,1).to(device)
        
        
        #using python nn
        self.Utransformation = nn.Linear(D,z_dim)
        self.Stransformation = nn.Linear(D,z_dim)
        self.Dectransformation = nn.Linear(z_dim,D)
        
        
        # ========================

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h)//h.shape[0]

    def encode(self, x):
        # TODO: Sample a latent vector z given an input x.
        # 1. Use the features extracted from the input to obtain mu and
        # log_sigma2 (mean and log variance) of the posterior p(z|x).
        # 2. Apply the reparametrization trick.
        # ====== YOUR CODE: ======
        h = self.features_encoder(x)
        
        h = h.view(h.shape[0],-1)
        
#         mu = torch.mm(h,self.Whu)
#         mu = mu + self.Bhu
        
        
#         log_sigma2 = torch.mm(h,self.Whs) + self.Bhs

        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        o = torch.zeros(1,self.z_dim)
        u = torch.normal(o,1).to(device)
        
        
        
        mu = self.Utransformation(h)
        log_sigma2 = self.Stransformation(h)
        
    
    
    
        z = mu + torch.exp(log_sigma2/2)*u
        
        # ========================

        return z, mu, log_sigma2

    def decode(self, z):
        # TODO: Convert a latent vector back into a reconstructed input.
        # 1. Convert latent to features.
        # 2. Apply features decoder.
        # ====== YOUR CODE: ======
#         h = torch.mm(z,self.Dec_T) + self.Dec_b
        h = self.Dectransformation(z)


        h = h.view(h.shape[0],-1,self.spatial,self.spatial)
        
        x_rec = self.features_decoder(h)
        # ========================

        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            # TODO: Sample from the model.
            # Generate n latent space samples and return their reconstructions.
            # Remember that for the model, this is like inference.
            # ====== YOUR CODE: ======
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            o = torch.zeros(n,self.z_dim)
            z = torch.normal(o,1).to(device)
            
            samples = self.decode(z)
            # ========================
        return samples.to('cpu')

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2
    
    
    def changeDevice(device):
        self.Whu = self.Whu.to(device)
        self.Whu = self.Whu.to(device)
        self.Whu = self.Whu.to(device)
        self.Whu = self.Whu.to(device)
        self.Whu = self.Whu.to(device)
        self.Whu = self.Whu.to(device)
        self.Whu = self.Whu.to(device)
        


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Pointwise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    loss, data_loss, kldiv_loss = None, None, None
    # TODO: Implement the VAE pointwise loss calculation.
    # Remember that the covariance matrix of the posterior is diagonal.
    # ====== YOUR CODE: ======
    
    N = x.shape[0]
    r = (x - xr).view(N,-1)
    D = r.shape[1]
    
    
    diff = torch.mm(r,r.transpose(0,1))
    data_loss_vec = torch.diag(diff) 
    
    data_loss = torch.mean(data_loss_vec) / D

    
    
    
    z_dim = z_mu.shape[1]
    
    # z_mu - (N , z_dim)
    # z log sigma - (N , z_dim)
    
    kldiv_loss_vec = z_log_sigma2.exp().sum(dim=1) + z_mu.norm(dim=1).pow(2) - z_dim - z_log_sigma2.sum(dim=1)
    kldiv_loss = torch.mean(kldiv_loss_vec) / z_dim
    
    
    
    
    loss = kldiv_loss + data_loss/ x_sigma2
    
    if loss > 10000 or torch.isnan(loss):
        raise AutoEncoderError('loss is Inf/Nan...')
    
    
    # ========================

    return loss, data_loss, kldiv_loss




class AutoEncoderError(Exception):
    def __init__(self, message):
        message = '################ ' + message + ' ################'
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
        self.message = message



