# Generate Emotional Landscape Images from Music

A two-stage deep learning system that analyzes emotional content in music and generates corresponding emotional landscape images using diffusion models.

## Overview

This project implements an end-to-end pipeline that:
1. Analyzes music to extract emotional features (Valence and Arousal values)
2. Generates landscape images that visually represent the detected emotions

**Valence-Arousal Model:**
- Valence: Emotional positivity (1=sad, 9=happy)
- Arousal: Emotional energy (1=calm, 9=energetic)

## System Architecture

### Stage 1: Music Emotion Analysis
- **Input:** Audio file (MP3, WAV, etc.)
- **Model:** Random Forest Regressor
- **Features:** Audio characteristics (MFCCs, spectral features, rhythm patterns)
- **Output:** Valence and Arousal values (1-9 scale)
- **Training Data:** DEAM + PMEmo datasets (combined ~3000 songs)

### Stage 2: Emotion-Conditioned Image Generation
- **Input:** Valence and Arousal values
- **Model:** Denoising Diffusion Probabilistic Model (DDPM) with UNet architecture
- **Conditioning:** VA values integrated through adaptive normalization and self-attention
- **Output:** 128x128 RGB landscape image
- **Training Data:** FindingEmo dataset (10,766 images) + CGnA10766 emotion annotations

## Working Models

**IMPORTANT: The following models are fully functional and production-ready:**

1. **Music Emotion Predictor**
   - Location: `Weights/Music/music_model_optimized.joblib`
   - Model: Random Forest with optimized hyperparameters
   - Training: DEAM + PMEmo combined dataset

2. **Diffusion Image Generator**
   - Location: `Weights/Diffusion/diffusion_epoch1650.pth` (recommended)
   - Architecture: Emotion-Conditioned UNet with self-attention
   - Training: 1650 epochs, batch size 16 (local training)
   - Performance: Best VA conditioning quality

**Note:** GAN-based models in `Image_Generation/GAN/` are experimental prototypes (WGAN, VAE, Progressive GAN, StyleGAN) and are not recommended for production use. The diffusion model demonstrates superior stability and emotion control.

## Requirements

```bash
Python 3.8+
torch>=2.0.0
torchvision
librosa>=0.10.0
scikit-learn
numpy
matplotlib
tqdm
joblib
```

## Installation

```bash
# Clone the repository
git clone https://github.com/WestCoastGod/Generate-Emotional-Landscape-Image-from-Music.git
cd Generate-Emotional-Landscape-Image-from-Music

# Install dependencies
pip install torch torchvision librosa scikit-learn numpy matplotlib tqdm joblib
```

## Usage

### Quick Start: Generate Image from Music

```python
# Run the main pipeline (to be implemented)
python main.py --audio path/to/music.mp3 --output generated_image.png
```

### Generate Image from Known VA Values

```python
from Image_Generation.Diffusion.generate import generate_emotion_image

# Generate a happy, energetic landscape (V=8.0, A=7.5)
generate_emotion_image(
    v=8.0,  # Valence: 1-9 scale
    a=7.5,  # Arousal: 1-9 scale
    model_path="Weights/Diffusion/diffusion_epoch1650.pth",
    save_path="output/happy_energetic_landscape.png",
    guidance_scale=5.0,  # Higher = stronger emotion effect
    seed=None  # Random generation (or set seed for reproducibility)
)
```

### Predict Music Emotions

```python
import joblib
import librosa
import numpy as np

# Load trained model
model = joblib.load("Weights/Music/music_model_optimized.joblib")

# Extract audio features (example - adapt from training notebook)
y, sr = librosa.load("path/to/music.mp3")
features = extract_audio_features(y, sr)  # Implement feature extraction

# Predict valence and arousal
valence, arousal = model.predict([features])[0]
print(f"Valence: {valence:.2f}, Arousal: {arousal:.2f}")
```

## Model Details

### Music Emotion Analysis
- **Dataset:** DEAM (2058 songs) + PMEmo (794 songs)
- **Features:** 
  - MFCCs (Mel-frequency cepstral coefficients)
  - Spectral features (centroid, rolloff, contrast)
  - Rhythm features (tempo, beat strength)
  - Zero-crossing rate, chroma features
- **Model:** Random Forest Regressor (optimized hyperparameters)
- **Performance:** See `Music_Emotion_Analysis/feature_importance.svg` for feature analysis

### Diffusion Image Generator
- **Architecture:** UNet with:
  - 5 encoder/decoder stages (64 → 128 → 256 → 512 → 1024 channels)
  - Self-attention at 16x16 resolution (2 layers)
  - VA conditioning via adaptive normalization (scale 0.3)
  - Classifier-free guidance for stronger emotion control
- **Training:**
  - Dataset: FindingEmo (10,766 landscape images) + CGnA10766 VA labels
  - Loss: MSE (denoising) + 0.1 * LPIPS (perceptual quality)
  - Epochs: 1650 (local model), batch size 16
  - Timesteps: 1000 (DDPM schedule)
- **Generation:** ~30 seconds per image (1000 denoising steps)
- **Guidance Scale:** 5.0 recommended (range 3-7)

### Datasets Used

1. **FindingEmo Dataset**
   - 10,766 landscape/nature photographs
   - Source: Emotional image dataset for computer vision research

2. **CGnA10766 Emotion Annotations**
   - Valence-Arousal labels for FindingEmo images
   - Human-annotated emotional ratings

3. **DEAM (Database for Emotional Analysis of Music)**
   - 2058 songs with continuous VA annotations
   - Multi-rater validated emotional labels

4. **PMEmo Dataset**
   - 794 popular music tracks
   - Detailed emotion annotations

## Project Structure

```
Generate-Emotional-Landscape-Image-from-Music/
├── main.py                          # Main entry script
├── README.md
├── .gitignore
│
├── Data/                            # All datasets
│   ├── Image/
│   │   ├── All_photos/             # Full image dataset
│   │   ├── Landscape/              # Landscape subset
│   │   ├── FindingEmo/             # FindingEmo dataset
│   │   ├── EmotionLabel/           # CGnA10766 VA annotations (CSV)
│   │   ├── analyze_dataset.py
│   │   └── dataset_*.png
│   └── Music/
│       ├── DEAM/                    # DEAM dataset
│       ├── PMEmo/                   # PMEmo dataset
│       └── EmotionLabel/
│           └── music_train_dataset.csv
│
├── Music_Emotion_Analysis/          # Music VA prediction
│   ├── music_data_clean_and_train.ipynb
│   ├── feature_importance.svg
│   └── predicted_vs_true_values.svg
│
├── Image_Generation/                # Image generation models
│   ├── GAN/                         # Legacy GAN experiments (not recommended)
│   │   ├── Models/
│   │   ├── Utils/
│   │   └── train_*.py
│   └── Diffusion/                   # Working diffusion model
│       ├── Utils/
│       │   └── dataloader.py        # Emotion dataset loader
│       ├── train_diffusion.py       # Local training (batch 16) - BEST
│       ├── train_diffusion_48gb.py  # Cloud training (batch 128)
│       └── generate.py              # Image generation script
│
├── Weights/                         # Trained model checkpoints
│   ├── Diffusion/
│   │   └── diffusion_epoch1650.pth # RECOMMENDED MODEL
│   ├── Music/
│   │   └── music_model_optimized.joblib
│   └── Backup/
│
└── Demo/                            # Sample outputs and documentation
    ├── GAN_Samples/
    ├── Diffusion_Samples/
    └── Others/
        ├── LOG.md
        ├── milestone_report.md
        └── PROJECT_HISTORY.md
```

## Training (Advanced)

### Retrain Music Model
See `Music_Emotion_Analysis/music_data_clean_and_train.ipynb` for the complete training pipeline.

### Retrain Diffusion Model

```bash
# Local training (recommended settings)
cd Image_Generation/Diffusion
python train_diffusion.py
```

**Training parameters:**
- Epochs: 1500-2000 recommended
- Batch size: 16 (for consumer GPUs)
- Learning rate: 0.0001
- Timesteps: 1000
- Hardware: NVIDIA GPU with 8GB+ VRAM

**Cloud training (48GB VRAM):**
- Batch size: 128 may cause mode collapse
- Requires 3000+ epochs to match local model performance
- Not recommended unless training time is critical

## Troubleshooting

### Generation produces grey/dark images
- Ensure using `diffusion_epoch1650.pth` or later checkpoint
- Try different `guidance_scale` values (3.0-7.0)
- Verify VA values are in correct range (1-9, not 0-1)

### Slow generation speed
- Reduce timesteps to 500 (faster, slightly lower quality)
- Use GPU if available
- Consider batch generation for multiple images

### Music model predictions seem off
- Verify audio file is properly loaded (correct sample rate)
- Check feature extraction matches training pipeline
- Ensure audio is at least 30 seconds long for reliable features

## Performance Notes

- **Generation time:** ~30 seconds per image (GPU), ~5 minutes (CPU)
- **Model size:** 420 MB (diffusion model), 1 MB (music model)
- **VRAM usage:** ~3 GB during generation
- **Training time:** ~12 hours for 1650 epochs (batch 16, single GPU)

## References

- **DDPM Paper:** Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- **FindingEmo Dataset:** 10,766 landscape photographs for emotion research
- **CGnA10766 Annotations:** Valence-Arousal labels for FindingEmo images
- **DEAM Dataset:** Database for emotional analysis of music (2058 songs)
- **PMEmo Dataset:** Popular music emotion dataset (794 tracks)

## Acknowledgments

- FindingEmo and CGnA10766 dataset creators
- DEAM and PMEmo dataset contributors
- PyTorch and related open-source communities
