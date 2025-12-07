# Generate Images from Music Emotion

A two-stage deep learning system that analyzes emotional content in music and generates corresponding images using diffusion models.

![progress](Src/progress.png)

![diffusion](Src/demo.png)
## Quick Start

```bash
# Install dependencies
pip install torch torchvision librosa scikit-learn numpy matplotlib tqdm joblib

# Generate landscape from music file
python main.py --audio path/to/music.mp3 --output generated_image.png

# Generate from valence/arousal values directly
python main.py --valence 8.0 --arousal 7.5 --output happy_energetic.png
```

## Pre-trained Weights

Download and place in `Weights/` folder:

| Model | Download Link |
|-------|---------------|
| Diffusion Model | [diffusion_epoch1650.pth](https://huggingface.co/spaces/WestCoastGod/photo-web-backend/resolve/main/music_to_image/models/diffusion_epoch1650.pth) |
| Music Emotion Model | [music_model_optimized.joblib](https://huggingface.co/spaces/WestCoastGod/photo-web-backend/resolve/main/music_to_image/models/music_model_optimized.joblib) |

## Datasets

| Dataset | Description | Link |
|---------|-------------|------|
| CGnA10766 | Image emotion annotations (Valence-Arousal) | [Figshare](https://figshare.com/articles/dataset/CGnA10766_Dataset/5383105) |
| DEAM | Music emotion dataset (2058 songs) | [CVML UNIGE](https://cvml.unige.ch/databases/DEAM/) |
| PMEmo | Music emotion dataset (794 songs) | [GitHub](https://github.com/HuiZhangDB/PMEmo?tab=readme-ov-file) |

## Usage

### Command Line

```bash
python main.py --help

Options:
  --audio, -a         Path to audio file for emotion analysis
  --valence, -v       Valence value (1-9, 1=sad, 9=happy)
  --arousal, -ar      Arousal value (1-9, 1=calm, 9=energetic)
  --output, -o        Output image path
  --guidance, -g      Guidance scale (default: 5.0)
  --seed, -s          Random seed for reproducibility
```

### Python API

```python
from Image_Generation.Diffusion.generate import generate_emotion_image

generate_emotion_image(
    v=8.0,  # Valence: 1-9 scale
    a=7.5,  # Arousal: 1-9 scale
    model_path="Weights/Diffusion/diffusion_epoch1650.pth",
    save_path="output/landscape.png",
    guidance_scale=5.0
)
```

## Requirements

- Python 3.8+
- PyTorch >= 2.0.0
- librosa >= 0.10.0
- scikit-learn, numpy, matplotlib, tqdm, joblib
