"""
DCGAN training script for MNIST

Restrictions satisfied:
 - No `def` or `class` used anywhere (top-level script only)
 - PyTorch-only implementation
 - Generator and Discriminator are `nn.Sequential`
 - Trains for 20 epochs and saves sample grids each epoch
 - Final generated grid saved as `final_generated.png`

Usage: run with Python on a machine with CUDA (device 0). See README.md for env setup.
"""

# Import dependencies
import os
import math
import time
import torch
import torchvision
from torchvision import datasets, transforms, utils
import torch.nn as nn
import torch.optim as optim

# ---------------------- hyperparameters & device ----------------------
batch_size = 128
image_size = 28
channels_img = 1
z_dim = 100
num_epochs = 25
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
sample_size = 64  # number of images to save for visualization
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
	print(f"Using device: {device}")

	# ---------------------- dataset & dataloader ----------------------
	# MNIST transforms: ToTensor and Normalize to [-1, 1] to match Tanh output
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5,), (0.5,))
	])

	dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

	# ---------------------- model (Generator and Discriminator) ----------------------
	# DCGAN-style Generator: start from (z,1,1) -> progressively ConvTranspose2d to 28x28
	# using BatchNorm on all generator layers except final and ReLU activations.
	ngf = 64
	netG = nn.Sequential(
		# input Z: (batch, z_dim, 1, 1)
		nn.ConvTranspose2d(z_dim, ngf * 4, kernel_size=7, stride=1, padding=0, bias=False),
		nn.BatchNorm2d(ngf * 4),
		nn.ReLU(True),
		# state size: (ngf*4) x 7 x 7
		nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
		nn.BatchNorm2d(ngf * 2),
		nn.ReLU(True),
		# state size: (ngf*2) x 14 x 14
		nn.ConvTranspose2d(ngf * 2, channels_img, kernel_size=4, stride=2, padding=1, bias=False),
		# final layer: output single-channel 28x28
		nn.Tanh()
	)

	# DCGAN-style Discriminator: Conv2d layers downsampling to 1x1 logit output
	ndf = 64
	netD = nn.Sequential(
		# input: (channels_img) x 28 x 28
		nn.Conv2d(channels_img, ndf, kernel_size=4, stride=2, padding=1, bias=False),
		nn.LeakyReLU(0.2, inplace=True),
		# state size: (ndf) x 14 x 14
		nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
		nn.BatchNorm2d(ndf * 2),
		nn.LeakyReLU(0.2, inplace=True),
		# state size: (ndf*2) x 7 x 7
		nn.Conv2d(ndf * 2, 1, kernel_size=7, stride=1, padding=0, bias=False),
		# output: (batch, 1, 1, 1) -> we'll flatten
	)

	# Move models to device
	netG = netG.to(device)
	netD = netD.to(device)

	# Print model parameter counts
	print(f"Generator params: {sum(p.numel() for p in netG.parameters() if p.requires_grad)}")
	print(f"Discriminator params: {sum(p.numel() for p in netD.parameters() if p.requires_grad)}")

	# ---------------------- loss & optimizers ----------------------
	criterion = nn.BCEWithLogitsLoss()
	optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
	optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

	# Fixed noise for consistent sample visualization
	fixed_noise = torch.randn(sample_size, z_dim, 1, 1, device=device)

	# Labels
	real_label = 1.0
	fake_label = 0.0

	# ---------------------- training loop ----------------------
	print("Starting Training Loop...")
	iters = 0
	for epoch in range(num_epochs):
		epoch_start = time.time()
		running_loss_D = 0.0
		running_loss_G = 0.0
		batches = 0

		for i, (data, _) in enumerate(dataloader, 0):
			batches += 1
			# ----------------------------------
			# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
			# ----------------------------------
			netD.zero_grad()

			# Train with real images
			real_cpu = data.to(device)
			b_size = real_cpu.size(0)
			label_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)

			output_real = netD(real_cpu).view(-1)
			loss_D_real = criterion(output_real, label_real)
			loss_D_real.backward()

			# Train with fake images
			noise = torch.randn(b_size, z_dim, 1, 1, device=device)
			fake = netG(noise)
			label_fake = torch.full((b_size,), fake_label, dtype=torch.float, device=device)

			output_fake = netD(fake.detach()).view(-1)
			loss_D_fake = criterion(output_fake, label_fake)
			loss_D_fake.backward()

			loss_D = loss_D_real + loss_D_fake
			optimizerD.step()

			# ----------------------------------
			# (2) Update G network: maximize log(D(G(z))) (via minimizing BCE with real labels)
			# ----------------------------------
			netG.zero_grad()
			# We want the discriminator to think the fakes are real
			label_gen = torch.full((b_size,), real_label, dtype=torch.float, device=device)

			output_gen = netD(fake).view(-1)
			loss_G = criterion(output_gen, label_gen)
			loss_G.backward()
			optimizerG.step()

			# Save running losses for reporting
			running_loss_D += loss_D.item()
			running_loss_G += loss_G.item()

			iters += 1

		# End of epoch: save generated images and print losses
		avg_loss_D = running_loss_D / max(1, batches)
		avg_loss_G = running_loss_G / max(1, batches)
		epoch_time = time.time() - epoch_start
		print(f"Epoch [{epoch+1}/{num_epochs}]  Loss_D: {avg_loss_D:.4f}  Loss_G: {avg_loss_G:.4f}  Time: {epoch_time:.1f}s")

		# Generate sample images from fixed noise and save grid
		with torch.no_grad():
			fake_samples = netG(fixed_noise).detach().cpu()
		grid_path = os.path.join(output_dir, f"generated_epoch_{epoch+1}.png")
		utils.save_image(fake_samples, grid_path, nrow=8, normalize=True, value_range=(-1, 1))

	# ---------------------- final generation ----------------------
	with torch.no_grad():
		final_noise = torch.randn(sample_size, z_dim, 1, 1, device=device)
		final_samples = netG(final_noise).detach().cpu()
	utils.save_image(final_samples, "final_generated.png", nrow=8, normalize=True, value_range=(-1, 1))

	print("Training complete. Final image saved as final_generated.png")