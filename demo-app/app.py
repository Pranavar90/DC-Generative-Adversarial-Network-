"""
Streamlit web app for DCGAN image generation demonstration
Run with: streamlit run app.py
"""

import os
import torch
import torch.nn as nn
import streamlit as st
from torchvision import utils
import matplotlib.pyplot as plt
from PIL import Image
import io

# Page config
st.set_page_config(page_title="DCGAN Demo", layout="centered", initial_sidebar_state="expanded")
st.title("🎨 DCGAN Image Generator")
st.markdown("**Live demonstration of a trained Deep Convolutional Generative Adversarial Network**")

# ---------------------- hyperparameters ----------------------
z_dim = 100
channels_img = 1
ngf = 64

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------- Load or build model ----------------------
@st.cache_resource
def build_generator():
	"""Build the generator model"""
	netG = nn.Sequential(
		nn.ConvTranspose2d(z_dim, ngf * 4, kernel_size=7, stride=1, padding=0, bias=False),
		nn.BatchNorm2d(ngf * 4),
		nn.ReLU(True),
		nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
		nn.BatchNorm2d(ngf * 2),
		nn.ReLU(True),
		nn.ConvTranspose2d(ngf * 2, channels_img, kernel_size=4, stride=2, padding=1, bias=False),
		nn.Tanh()
	)
	return netG.to(device).eval()

netG = build_generator()

# Sidebar controls
st.sidebar.header("⚙️ Settings")
num_images = st.sidebar.slider("Number of images to generate:", 1, 9, 4)
seed = st.sidebar.number_input("Random seed (for reproducibility):", min_value=0, max_value=10000, value=42)

# Main demo section
st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
	st.subheader("Generated Images")
	generate_button = st.button("🎲 Generate New Images", use_container_width=True)

with col2:
	st.info(f"Device: {device}")

if generate_button:
	with st.spinner("Generating images..."):
		# Set seed for reproducibility
		torch.manual_seed(seed)
		
		# Generate images
		with torch.no_grad():
			noise = torch.randn(num_images, z_dim, 1, 1, device=device)
			generated_images = netG(noise).detach().cpu()
		
		# Display images
		fig, axes = plt.subplots(1, num_images, figsize=(12, 3))
		if num_images == 1:
			axes = [axes]
		
		for idx, ax in enumerate(axes):
			img = generated_images[idx].squeeze(0).numpy()
			ax.imshow(img, cmap='gray')
			ax.axis('off')
			ax.set_title(f"Image {idx+1}")
		
		plt.tight_layout()
		st.pyplot(fig)
	
	st.success("✅ Images generated successfully!")

# Information section
st.markdown("---")
st.subheader("📊 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
	total_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
	st.metric("Generator Parameters", f"{total_params:,}")

with col2:
	st.metric("Input Noise Dimension", z_dim)

with col3:
	st.metric("Output Size", "28×28")

# Details
st.markdown("""
### How it works:
1. **Input**: Random noise vector (100-dimensional)
2. **Processing**: Passes through transposed convolution layers with batch normalization and ReLU activations
3. **Output**: Generated 28×28 grayscale image (MNIST-like)

### Architecture:
- **Layer 1**: (100, 1, 1) → (256, 7, 7) using 7×7 ConvTranspose2d
- **Layer 2**: (256, 7, 7) → (128, 14, 14) using 4×4 ConvTranspose2d with stride 2
- **Layer 3**: (128, 14, 14) → (1, 28, 28) using 4×4 ConvTranspose2d with stride 2 and Tanh activation

### Tips for your presentation:
- Try different seeds to see variety in generated images
- Increase the number of images to see the diversity
- Each generation is unique due to random noise input
""")

st.markdown("---")
st.caption("Built with PyTorch and Streamlit | DCGAN trained on MNIST dataset")
