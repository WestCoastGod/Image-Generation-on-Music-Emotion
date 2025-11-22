# AIST4010 Project Milestone Report

**Generate Images from Music Emotion**

**Author:** Zhang Xian (1155193168)

**Date:** November 4, 2025

---

## 1. Introduction (20%)

Music and images both express emotions powerfully. People naturally imagine scenes when listening to music, but computer systems that automatically create matching images are still limited. This project generates landscape images that represent music emotions, bridging audio and visual expression.

This problem has practical applications like automatic album cover creation, music visualization for streaming platforms, and mood-based environments. We use the valence-arousal (VA) model where valence measures positive/negative feelings and arousal measures calm/excited states. This lets computers process emotions as numbers. The project combines two approaches: predicting emotions from music and generating images from those predictions.

Previous research treated music emotion recognition and image generation separately. Our project connects music emotions with realistic landscape images, which is challenging because landscapes express emotions through colors, lighting, and composition. Our two-stage method uses (1) Random Forest to predict VA values from music features, and (2) progressive GAN with Emotional Residual Units to generate 128×128 landscape images.

## 2. Related Work (25%)

### Music Emotion Recognition
Early work by Yang et al. predicted valence and arousal from audio features, establishing dimensional emotion prediction [1]. The DEAM dataset [2] and PMEmo dataset [3] provided labeled music for training emotion recognition models. Research shows that combining multiple audio features (MFCCs, spectral features, tempo) improves prediction accuracy [4]. We extract 148 features from each song using librosa.

### Generative Adversarial Networks
Goodfellow et al. introduced GANs where a generator and discriminator compete to create realistic images [5]. WGAN-GP improved training stability through gradient penalty, which we use in our discriminator [6]. Karras et al. developed progressive growing GANs that start at 4×4 resolution and gradually increase to higher resolutions, preventing training problems [7]. Our generator follows this approach, progressing from 4×4 to 128×128.

### Emotion-Conditioned Generation
Park et al. specifically addressed emotional landscape generation using GANs conditioned on valence and arousal [8]. They proposed Affective Feature Matching (AFM) loss to maintain emotional consistency between real and generated images. Our work builds on this foundation, combining AFM loss with WGAN-GP for stable training. Other work explored music-to-abstract-art [9] and cross-modal learning [10], showing that meaningful connections between audio and visual data are possible.

### Evaluation
FID (Fréchet Inception Distance) measures image quality by comparing distributions of real and generated images [11]. We plan to use FID as our primary evaluation metric alongside user studies for subjective assessment.

## 3. Data (15%)

### Music Datasets
We use two datasets with ~2,800 songs total. **DEAM** contains 2,058 songs (45-second clips) with averaged valence and arousal labels from multiple raters. **PMEmo** has 794 Chinese pop songs (30-second chorus clips) with VA annotations, adding musical diversity. 

We extract 148 features per song using librosa: 13 MFCCs, 12 chroma features, spectral contrast, tempo, loudness, and others. Each feature's mean and standard deviation over time creates the feature vector, normalized to 0-1 range.

### Image Dataset
**CGnA10766** contains 10,766 landscape photos (mountains, forests, beaches, skies) with VA ratings on a 1-9 scale. Images are resized to 128×128 and normalized to -1 to 1 range. Progressive training starts at 4×4 resolution.

**Data Issues:** Both datasets have fewer examples of extreme negative emotions. Landscape photos skew toward positive emotions since people prefer photographing pleasant scenes. Data cleaning removed corrupted files.

## 4. Approach (20%)

### Stage 1: Music Emotion Prediction
We use **Random Forest** to predict valence and arousal from 148 extracted audio features. Random Forest handles high-dimensional data well and captures complex patterns. The process: load audio → extract features → calculate mean/std → normalize to 0-1 range. We start with default parameters and plan optimization later.

### Stage 2: Image Generation
We use **Progressive GAN** conditioned on VA values.

**Generator:** Takes 100 random numbers + 2 emotion values as input. Starts at 4×4 resolution and progresses through six stages to 128×128. **Emotional Residual Units (ERU)** let VA values influence generation at each resolution level. Output uses Tanh for -1 to 1 range.

**Discriminator:** Mirrors the generator in reverse. Takes images and VA values, outputs real/fake scores and intermediate features for AFM loss.

**Training:** Progressive stages add resolution every 10 epochs. Uses WGAN-GP (gradient penalty weight=10) for stability and AFM loss (weight=100) for emotional consistency. Total loss: L = L_adversarial + L_gradient_penalty + 100(L_valence + L_arousal). Adam optimizer (lr=0.001), batch size 16, 1711 epochs.

**Baselines:** Plan to compare against (1) unconditional GAN, (2) non-progressive GAN, (3) GAN without AFM loss.

## 5. Preliminary Results (20%)

### Music Emotion Prediction
Random Forest achieved **MSE: 0.832** and **R² Score: 0.407** on combined datasets, explaining 41% of variance in VA values. While moderate, this provides functional emotion predictions for image generation. The model predicts arousal better than valence, matching prior research. Performance is limited by: (1) subjective emotion ratings, (2) averaged labels missing individual differences, (3) need for parameter tuning, (4) potential benefits from deep learning approaches.

### Image Generation
Progressive GAN trained successfully through all stages without collapse. Generated 128×128 images show recognizable landscapes (sky, horizon, terrain). Colors vary with VA values—higher valence produces warmer, brighter scenes; lower arousal creates calmer compositions. Training observations:
- 4×4 stage learned colors quickly
- Smooth transitions to higher resolutions
- AFM loss maintained emotional consistency
- WGAN-GP prevented collapse

**Limitations:** Images are less sharp than real photos; some VA values produce similar outputs; limited diversity; extreme negative emotions produce unclear results due to dataset imbalance.

### End-to-End System
The complete pipeline works: music → feature extraction → VA prediction → image generation (128×128). This demonstrates feasibility.

### Next Steps
**Plans:** (1) Optimize Random Forest parameters, (2) Test longer GAN training with different AFM weights, (3) Calculate FID scores, (4) Design user study for subjective evaluation, (5) Train baseline models for comparison, (6) Explore 256×256 resolution.

**Challenges:** Limited GPU access slows training; dataset imbalance toward positive emotions; parameter sensitivity requires careful tuning; subjective evaluation without ground truth.

---

## References

[1] Yang, Y. H., et al. (2008). A regression approach to music emotion recognition. *IEEE Transactions on Audio, Speech, and Language Processing*. https://ieeexplore.ieee.org/document/4432703

[2] Aljanaki, A., Yang, Y. H., & Soleymani, M. (2017). Developing a benchmark for emotional analysis of music. *PloS one*. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0173392

[3] Zhang, K., et al. (2018). The PMEmo dataset for music emotion recognition. *ACM Multimedia*. https://github.com/HuiZhangDB/PMEmo

[4] Panda, R., et al. (2020). Audio features for music emotion recognition. *IEEE Transactions on Affective Computing*. https://ieeexplore.ieee.org/document/8848053

[5] Goodfellow, I., et al. (2014). Generative adversarial nets. *NeurIPS*. https://arxiv.org/abs/1406.2661

[6] Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein GAN. *ICML*. https://arxiv.org/abs/1701.07875

[7] Karras, T., et al. (2018). Progressive growing of GANs. *ICLR*. https://arxiv.org/abs/1710.10196

[8] Park, S., et al. (2020). Emotional landscape image generation using GANs. *ACCV*. https://openaccess.thecvf.com/content/ACCV2020/papers/Park_Emotional_Landscape_Image_Generation_Using_Generative_Adversarial_Networks_ACCV_2020_paper.pdf

[9] Mehta, D., et al. (2018). Music-to-abstract image generation. *Proceedings*. https://arxiv.org/abs/1811.00658

[10] Chen, K., et al. (2020). Music-to-dance generation. *arXiv preprint*. https://arxiv.org/abs/2002.03761

[11] Heusel, M., et al. (2017). GANs trained by a two time-scale update rule. *NeurIPS*. https://arxiv.org/abs/1706.08500

---

*Note: This milestone report represents work in progress. Final results and comprehensive evaluations will be presented in the final report.*
