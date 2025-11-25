# DC-GAN: Deep Convolutional Generative Adversarial Network

A PyTorch implementation of Deep Convolutional Generative Adversarial Network (DCGAN) trained on the MNIST dataset to generate handwritten digit images.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Overview

This project implements a DCGAN architecture that learns to generate realistic handwritten digits by training on the MNIST dataset. The implementation follows the original DCGAN paper's architecture guidelines and uses PyTorch for deep learning operations.

### Key Features

- ✅ Pure PyTorch implementation using `nn.Sequential`
- ✅ DCGAN architecture with transposed convolutions
- ✅ Trains on MNIST dataset (28×28 grayscale images)
- ✅ CUDA/GPU support for accelerated training
- ✅ Automatic sample generation during training
- ✅ Interactive Streamlit web demo for live generation

## 🏗️ Architecture

### Generator
The generator transforms a 100-dimensional noise vector into a 28×28 grayscale image through the following layers:

```
Input: (100, 1, 1) random noise
  ↓ ConvTranspose2d (7×7 kernel)
(256, 7, 7) + BatchNorm + ReLU
  ↓ ConvTranspose2d (4×4 kernel, stride=2)
(128, 14, 14) + BatchNorm + ReLU
  ↓ ConvTranspose2d (4×4 kernel, stride=2)
Output: (1, 28, 28) + Tanh
```

### Discriminator
The discriminator classifies images as real or fake through convolutional downsampling:

```
Input: (1, 28, 28) image
  ↓ Conv2d (4×4 kernel, stride=2)
(64, 14, 14) + LeakyReLU
  ↓ Conv2d (4×4 kernel, stride=2)
(128, 7, 7) + BatchNorm + LeakyReLU
  ↓ Conv2d (7×7 kernel)
Output: (1, 1, 1) logit
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster training)
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Pranavar90/DC-Generative-Adversarial-Network-.git
cd DC-Generative-Adversarial-Network-
```

2. **Install dependencies**
```bash
pip install torch torchvision
pip install streamlit matplotlib pillow  # For demo app
```

### Required Packages

- `torch` >= 2.0.0
- `torchvision` >= 0.15.0
- `streamlit` >= 1.28.0 (for demo app)
- `matplotlib` >= 3.5.0 (for demo app)
- `pillow` >= 9.0.0 (for demo app)

## 💻 Usage

### Training the Model

Run the main training script:

```bash
python dcgan.py
```

**Training Parameters:**
- Batch size: 128
- Epochs: 25
- Learning rate: 0.0002
- Optimizer: Adam (β₁=0.5, β₂=0.999)
- Latent dimension: 100

**Output:**
- Generated image grids saved in `outputs/` directory after each epoch
- Final generated image saved as `final_generated.png`
- Training progress printed to console

### Running the Demo App

Launch the interactive Streamlit web application:

```bash
cd demo-app
streamlit run app.py
```

The demo app allows you to:
- Generate new images with a single click
- Control the number of images to generate (1-9)
- Set random seed for reproducible results
- View model architecture and parameters

## 📁 Project Structure

```
dcgan/
├── dcgan.py              # Main training script
├── final_generated.png   # Final output from training
├── outputs/              # Generated images per epoch
│   ├── generated_epoch_1.png
│   ├── generated_epoch_2.png
│   └── ...
├── data/                 # MNIST dataset (auto-downloaded)
└── demo-app/            # Streamlit web application
    ├── app.py           # Demo app script
    └── README.md        # Demo app documentation
```

## 🎯 Results

The model generates realistic handwritten digits after training for 25 epochs. Sample outputs show diverse digit styles and variations learned from the MNIST dataset.

### Training Progress

- **Epoch 1-5**: Basic digit shapes emerge
- **Epoch 6-15**: Clearer digit formations with better details
- **Epoch 16-25**: High-quality, diverse digit generation

## 🔧 Hyperparameters

You can modify these parameters in `dcgan.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 128 | Training batch size |
| `num_epochs` | 25 | Number of training epochs |
| `lr` | 0.0002 | Learning rate for both G and D |
| `z_dim` | 100 | Latent noise dimension |
| `ngf` | 64 | Generator feature map size |
| `ndf` | 64 | Discriminator feature map size |

## 📊 Model Details

- **Generator Parameters**: ~1.4M trainable parameters
- **Discriminator Parameters**: ~0.5M trainable parameters
- **Training Time**: ~10-15 minutes on CUDA GPU (varies by hardware)
- **Dataset**: MNIST (60,000 training images)

## 🎓 References

This implementation is based on the DCGAN paper:

> Radford, A., Metz, L., & Chintala, S. (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*. arXiv preprint arXiv:1511.06434.

## 📝 Implementation Notes

- No `def` or `class` keywords used in the main training script (procedural style)
- Uses `nn.Sequential` for both Generator and Discriminator
- Follows DCGAN architecture guidelines (BatchNorm, LeakyReLU, etc.)
- Automatic MNIST dataset download on first run
- Images normalized to [-1, 1] range to match Tanh output

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Pranavar90**

- GitHub: [@Pranavar90](https://github.com/Pranavar90)
- Repository: [DC-Generative-Adversarial-Network-](https://github.com/Pranavar90/DC-Generative-Adversarial-Network-)

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- MNIST dataset creators
- Original DCGAN paper authors

---

⭐ If you find this project helpful, please consider giving it a star!
