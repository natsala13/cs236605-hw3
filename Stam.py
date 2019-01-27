import torch
import hw3.autoencoder as autoencoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

encoder = autoencoder.EncoderCNN(in_channels=3, out_channels=256)
decoder = autoencoder.DecoderCNN(in_channels=256, out_channels=3)
vae = autoencoder.VAE(encoder, decoder, (3,64,64), 5)

vae.cuda()

print(vae.Whu.is_cuda)


vae.Whu = vae.Whu.to(device)

print(vae.Whu.is_cuda)
print(vae.Whs.is_cuda)

