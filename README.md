# Conditional GAN for Animal Image Generation

## 📌 Overview
This project implements a **Conditional Deep Convolutional GAN (cGAN)** to generate synthetic animal images from the **Animals-10 dataset**. The model is conditioned on class labels, enabling class-specific image generation (e.g., cats, dogs, butterflies).

## ⚙️ Methodology
- **Dataset:** Animals-10 (10 animal classes, e.g., cavallo, gatto, cane, farfalla).
- **Preprocessing:** Images resized to 64×64, normalized to [-1, 1], batched (128), GPU-enabled.
- **Architecture:**
  - **Generator:** Latent vector + class embedding → transposed conv layers → 64×64 image.
  - **Discriminator:** Image + label embedding → conv layers → scalar score.
- **Training:** Hinge loss, Adam optimizer (lr=0.0001), alternating updates, 5 epochs (initial).

## 📊 Results
- Stable training, no mode collapse.
- Early outputs show **class-aware color patterns** (e.g., butterflies → yellow/green, cats → brown/white).
- Images are blurry/noisy at 5 epochs but demonstrate label-conditioned learning.

## 🚧 Limitations
- Only 5 epochs → insufficient for high-quality results.
- Low resolution (64×64).
- No quantitative evaluation (FID, IS).
- Blurry outputs, limited fine details.

## 🔮 Potential Improvements
- Train longer (20–50 epochs).
- Add **spectral normalization**, **attention layers**, or switch to **WGAN-GP**.
- Increase resolution to 128×128.
- Compute FID/IS for objective evaluation.
- Build interactive UI for class-wise generation.

## ✅ Conclusion
This project demonstrates a working **cGAN pipeline** for class-conditioned animal image generation. While early results are blurry, the model shows promising learning dynamics and sets the foundation for scaling to higher resolution and advanced generative models.
