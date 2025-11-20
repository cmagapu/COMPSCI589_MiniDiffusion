# Mini Diffusion Model for MNIST

A lightweight implementation of Denoising Diffusion Probabilistic Models (DDPM) for MNIST digit generation, exploring optimization techniques including pruning and architectural modifications.

## Overview

This project investigates how diffusion models can be simplified for small-scale tasks while maintaining generation quality. We implement a baseline DDPM with a simplified U-Net architecture and compare its performance against optimized variants using structured/unstructured pruning and architectural modifications.

**Key Features:**
- Minimal DDPM implementation for educational purposes
- Comprehensive optimization strategies (pruning, architecture reduction)
- Extensive evaluation metrics (Pseudo-FID, SSIM, efficiency benchmarks)
- Self-contained Jupyter notebook - everything in one place!

## Results Summary

| Model | Parameters | Size (MB) | Pseudo-FID ↓ | SSIM ↑ | Inference (s) |
|-------|-----------|-----------|--------------|--------|---------------|
| Baseline | 2.53M | 9.72 | 10.75 | 0.3475 | 57.6 |
| Pruned | 2.53M | 9.72 | 2.77 | 0.3674 | 57.6 |
| Reduced Channels | 634K | 2.5 | **1.76** | **0.3688** | 31.3 |
| Shallow Network | 2.15M | 8.2 | 2.39 | 0.3659 | 81.6 |

**Key Findings:**
- **75% parameter reduction** achieved with reduced channels architecture
- **83% quality improvement** (FID: 10.75 → 1.76) with architectural optimization
- Architectural redesign significantly outperforms traditional pruning approaches
- Smaller models can outperform larger ones on simple tasks with proper design

## Architecture

### Baseline Model: Simplified U-Net

The core architecture uses a U-Net with:
- **Time embeddings**: Sinusoidal position embeddings with MLP projection
- **Encoder**: Convolutional blocks (32, 64, 128 channels) with downsampling
- **Bottleneck**: Two residual blocks for feature processing
- **Decoder**: Transposed convolutions with skip connections from encoder
- **Noise prediction**: Final 1×1 conv layer outputs predicted noise

### Diffusion Process

**Forward process** (noise addition):
```
q(x_t | x_0) = N(x_t; √(ᾱ_t)x_0, (1 - ᾱ_t)I)
```

**Reverse process** (denoising):
```
x_{t-1} = 1/√(α_t) * [x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t, t)] + σ_t * z
```

**Training objective**: MSE loss between predicted and actual noise
```
L = E_{t,x_0,ε} ||ε - ε_θ(√(ᾱ_t)x_0 + √(1-ᾱ_t)ε, t)||²
```

## Installation

```bash
# Clone the repository
git clone https://github.com/cmagapu/mini-diffusion-mnist.git
cd mini-diffusion-mnist

# Install dependencies
pip install torch torchvision numpy matplotlib scikit-image scipy
```

## Usage

Simply open and run the Jupyter notebook:

```bash
jupyter notebook diffusion_model.ipynb
```

The notebook contains all the code for:
- Training the baseline DDPM
- Applying pruning (structured + unstructured)
- Training architectural variants (reduced channels, shallow network)
- Fine-tuning all models
- Generating samples
- Comprehensive evaluation with all metrics


## Notebook Structure

The notebook is organized into the following sections:

1. **Setup & Imports** - Dependencies and utility functions
2. **U-Net Architecture** - Simplified U-Net with time embeddings
3. **DDPM Implementation** - Forward/reverse diffusion processes
4. **Baseline Training** - Train baseline model (50 epochs)
5. **Pruning** - Structured (10%) + Unstructured (30%) pruning
6. **Fine-tuning Pruned Model** - Recover performance (20 epochs)
7. **Architectural Variants** - Reduced channels & shallow network
8. **Evaluation** - SSIM, Pseudo-FID, efficiency metrics
9. **Visualization** - Generated samples and comparisons

## Optimization Techniques

### 1. Pruning

**Structured Pruning (10%):**
- Removes entire filters/channels from convolutional layers
- Applied using L1 magnitude-based selection
- Enables faster inference on standard hardware

**Unstructured Pruning (30%):**
- Removes individual low-magnitude weights
- Applied across all convolutional and linear layers
- Maximizes sparsity for size reduction

**Fine-tuning:**
- 20 epochs with reduced learning rate (1e-4)
- Recovers performance degradation from pruning

### 2. Architectural Modifications

**Reduced Channels:**
- Channels: (16, 32, 64) vs baseline (32, 64, 128)
- **75% parameter reduction** (634K vs 2.53M)
- Requires fine-tuning (10 epochs, lr=1e-5) to overcome mode collapse

**Shallow Network:**
- 2 levels instead of 3: (64, 128) channels
- **15% parameter reduction** (2.15M vs 2.53M)
- Fine-tuned for 20 epochs (lr=1e-4)

## Evaluation Metrics

### Efficiency Metrics
- **Parameters**: Total trainable parameters
- **Model Size**: Disk space (MB)
- **Inference Latency**: Time to generate 10,000 samples

### Quality Metrics
- **SSIM**: Structural similarity to real images (higher is better)
- **Pseudo-FID**: Feature-based quality/diversity measure (lower is better)
  - Uses custom-trained MNIST classifier (98.7% accuracy)
- **Digit Distribution**: Balance across 10 classes (0-9)

## Key Insights

1. **Architecture > Pruning**: Architectural redesign achieves better efficiency-quality trade-offs than post-training compression

2. **Task-Appropriate Scaling**: Smaller, well-designed models can outperform larger ones on simple tasks like MNIST

3. **Fine-tuning Critical**: Architectural variants require specialized fine-tuning to overcome initial mode collapse

4. **Metrics vs. Perception**: Trade-off exists between quantitative metrics (FID) and visual sharpness - baseline generates crisper digits despite worse FID

## Limitations & Future Work

**Current Limitations:**
- Pruning implementation didn't achieve expected size reductions
- Architectural variants show higher inference latency in current setup
- MNIST is a simple dataset - findings may not generalize to complex images

**Future Directions:**
- Explore architectural principles on CIFAR-10, CelebA
- Investigate quantization for further compression
- Study relationship between quantitative metrics and perceptual quality
- Implement more efficient sampling algorithms (DDIM, DPM-Solver)
- Add conditional generation capabilities


## References

1. Ho et al. (2020). "Denoising Diffusion Probabilistic Models." NeurIPS.
2. Dhariwal & Nichol (2021). "Diffusion Models Beat GANs on Image Synthesis." NeurIPS.
3. Ronneberger et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation."
4. Zhang et al. (2024). "Effortless Efficiency: Low-Cost Pruning of Diffusion Models."

## License

MIT License - see LICENSE file for details

## Authors

- **Chandana Magapu** - *UMass Amherst* - hmagapu@umass.edu
- **Aproorva Jaiswal** - *UMass Amherst* - ajaiswal@umass.edu
