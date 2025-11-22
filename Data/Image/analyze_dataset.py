"""
Dataset Analysis Script
Analyzes landscape images and VA labels to understand data distribution
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import seaborn as sns

def analyze_dataset(img_dir, label_csv):
    """Comprehensive analysis of the landscape dataset"""
    
    print("="*60)
    print("LANDSCAPE DATASET ANALYSIS")
    print("="*60)
    
    # Load VA labels
    df = pd.read_csv(label_csv)
    print(f"\n1. DATASET SIZE")
    print(f"   Total images in CSV: {len(df)}")
    
    # Check which images actually exist
    existing_images = []
    missing_images = []
    
    for idx, row in df.iterrows():
        # Convert image ID to filename with .jpg extension
        img_id = int(row['image'])
        img_name = f"{img_id}.jpg"
        
        img_path = os.path.join(img_dir, img_name)
        if os.path.exists(img_path):
            existing_images.append((img_path, row['valence'], row['arousal']))
        else:
            missing_images.append(img_name)
    
    print(f"   Images found on disk: {len(existing_images)}")
    if missing_images:
        print(f"   Missing images: {len(missing_images)}")
        print(f"   First 5 missing: {missing_images[:5]}")
    
    # Extract VA values for existing images
    valences = [v for _, v, _ in existing_images]
    arousals = [a for _, _, a in existing_images]
    
    # 2. VA Value Statistics
    print(f"\n2. VALENCE-AROUSAL STATISTICS")
    print(f"   Valence range: [{min(valences):.2f}, {max(valences):.2f}]")
    print(f"   Valence mean: {np.mean(valences):.2f} ± {np.std(valences):.2f}")
    print(f"   Arousal range: [{min(arousals):.2f}, {max(arousals):.2f}]")
    print(f"   Arousal mean: {np.mean(arousals):.2f} ± {np.std(arousals):.2f}")
    
    # Normalized values (what model sees)
    v_norm = [(v - 1) / 8 for v in valences]  # Assuming 1-9 scale
    a_norm = [(a - 1) / 8 for a in arousals]
    print(f"\n   After normalization [0, 1]:")
    print(f"   Valence: [{min(v_norm):.3f}, {max(v_norm):.3f}], mean={np.mean(v_norm):.3f}")
    print(f"   Arousal: [{min(a_norm):.3f}, {max(a_norm):.3f}], mean={np.mean(a_norm):.3f}")
    
    # 3. Image Resolution Analysis
    print(f"\n3. IMAGE RESOLUTION ANALYSIS")
    print(f"   Analyzing {min(len(existing_images), 100)} sample images...")
    
    resolutions = []
    aspect_ratios = []
    file_sizes = []
    
    sample_size = min(len(existing_images), 100)
    for img_path, _, _ in existing_images[:sample_size]:
        try:
            img = Image.open(img_path)
            width, height = img.size
            resolutions.append((width, height))
            aspect_ratios.append(width / height)
            file_sizes.append(os.path.getsize(img_path) / 1024)  # KB
            img.close()
        except Exception as e:
            print(f"   Error loading {img_path}: {e}")
    
    if resolutions:
        widths = [w for w, h in resolutions]
        heights = [h for w, h in resolutions]
        
        print(f"\n   Original Resolutions:")
        print(f"   Width:  min={min(widths)}px, max={max(widths)}px, mean={int(np.mean(widths))}px")
        print(f"   Height: min={min(heights)}px, max={max(heights)}px, mean={int(np.mean(heights))}px")
        print(f"   Aspect ratio: {min(aspect_ratios):.2f} - {max(aspect_ratios):.2f} (mean={np.mean(aspect_ratios):.2f})")
        print(f"   File size: {min(file_sizes):.1f} - {max(file_sizes):.1f} KB (mean={np.mean(file_sizes):.1f} KB)")
        
        # Common resolutions
        unique_res = {}
        for res in resolutions:
            unique_res[res] = unique_res.get(res, 0) + 1
        
        print(f"\n   Most common resolutions:")
        for res, count in sorted(unique_res.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {res[0]}x{res[1]}: {count} images")
        
        # Calculate resize impact
        print(f"\n4. RESIZE IMPACT (to 128x128)")
        for res, count in sorted(unique_res.items(), key=lambda x: x[1], reverse=True)[:3]:
            w, h = res
            if w > h:
                new_h = 128
                new_w = int(w * 128 / h)
                crop = (new_w - 128) // 2
                print(f"   {w}x{h} → {new_w}x{new_h} → crop {crop}px from sides → 128x128")
            else:
                new_w = 128
                new_h = int(h * 128 / w)
                crop = (new_h - 128) // 2
                print(f"   {w}x{h} → {new_w}x{new_h} → crop {crop}px from top/bottom → 128x128")
    
    # 5. Create Visualizations
    print(f"\n5. GENERATING VISUALIZATIONS...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # VA Distribution (2D scatter)
    ax1 = plt.subplot(2, 3, 1)
    scatter = ax1.scatter(valences, arousals, alpha=0.5, s=20, c=valences, cmap='RdYlGn')
    ax1.set_xlabel('Valence (1-9)')
    ax1.set_ylabel('Arousal (1-9)')
    ax1.set_title(f'VA Distribution (n={len(existing_images)})')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Valence')
    
    # Valence histogram
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(valences, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(valences), color='red', linestyle='--', label=f'Mean={np.mean(valences):.2f}')
    ax2.set_xlabel('Valence')
    ax2.set_ylabel('Count')
    ax2.set_title('Valence Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Arousal histogram
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(arousals, bins=30, color='orange', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(arousals), color='red', linestyle='--', label=f'Mean={np.mean(arousals):.2f}')
    ax3.set_xlabel('Arousal')
    ax3.set_ylabel('Count')
    ax3.set_title('Arousal Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Normalized VA distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(v_norm, a_norm, alpha=0.5, s=20, c=v_norm, cmap='RdYlGn')
    ax4.set_xlabel('Normalized Valence [0, 1]')
    ax4.set_ylabel('Normalized Arousal [0, 1]')
    ax4.set_title('Normalized VA Distribution (Model Input)')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-0.1, 1.1)
    ax4.set_ylim(-0.1, 1.1)
    
    # Resolution distribution
    if resolutions:
        ax5 = plt.subplot(2, 3, 5)
        ax5.scatter(widths, heights, alpha=0.5, s=20, color='purple')
        ax5.set_xlabel('Width (px)')
        ax5.set_ylabel('Height (px)')
        ax5.set_title('Image Resolutions')
        ax5.grid(True, alpha=0.3)
        
        # Target resolution box
        ax5.axhline(128, color='red', linestyle='--', alpha=0.5, label='Target: 128x128')
        ax5.axvline(128, color='red', linestyle='--', alpha=0.5)
        ax5.legend()
    
    # Aspect ratio distribution
    if aspect_ratios:
        ax6 = plt.subplot(2, 3, 6)
        ax6.hist(aspect_ratios, bins=20, color='teal', alpha=0.7, edgecolor='black')
        ax6.axvline(1.0, color='red', linestyle='--', label='Square (1:1)')
        ax6.set_xlabel('Aspect Ratio (W/H)')
        ax6.set_ylabel('Count')
        ax6.set_title('Aspect Ratio Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to same directory as script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'dataset_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved visualization to: {output_path}")
    
    # 6. VA Quadrant Analysis
    print(f"\n6. VA QUADRANT DISTRIBUTION")
    v_mid = (max(valences) + min(valences)) / 2
    a_mid = (max(arousals) + min(arousals)) / 2
    
    q1 = sum(1 for v, a in zip(valences, arousals) if v >= v_mid and a >= a_mid)
    q2 = sum(1 for v, a in zip(valences, arousals) if v < v_mid and a >= a_mid)
    q3 = sum(1 for v, a in zip(valences, arousals) if v < v_mid and a < a_mid)
    q4 = sum(1 for v, a in zip(valences, arousals) if v >= v_mid and a < a_mid)
    
    print(f"   High V, High A (Happy/Excited):  {q1} ({q1/len(valences)*100:.1f}%)")
    print(f"   Low V, High A (Angry/Tense):     {q2} ({q2/len(valences)*100:.1f}%)")
    print(f"   Low V, Low A (Sad/Depressed):    {q3} ({q3/len(valences)*100:.1f}%)")
    print(f"   High V, Low A (Calm/Relaxed):    {q4} ({q4/len(valences)*100:.1f}%)")
    
    # 7. Recommendations
    print(f"\n7. RECOMMENDATIONS")
    
    if len(existing_images) < 1000:
        print(f"   ⚠ Dataset size ({len(existing_images)}) is small for 128x128 GANs")
        print(f"     Recommended: 2000+ images for stable training")
    elif len(existing_images) < 2000:
        print(f"   ✓ Dataset size ({len(existing_images)}) is acceptable for 128x128")
        print(f"     Consider data augmentation for better results")
    else:
        print(f"   ✓ Dataset size ({len(existing_images)}) is good for 128x128 training")
    
    v_coverage = (max(v_norm) - min(v_norm))
    a_coverage = (max(a_norm) - min(a_norm))
    
    if v_coverage < 0.8 or a_coverage < 0.8:
        print(f"   ⚠ VA coverage is limited (V: {v_coverage:.2f}, A: {a_coverage:.2f})")
        print(f"     Some emotions may be underrepresented")
    else:
        print(f"   ✓ Good VA coverage (V: {v_coverage:.2f}, A: {a_coverage:.2f})")
    
    if max(widths) > 512 or max(heights) > 512:
        print(f"   ℹ Some images are high resolution (up to {max(widths)}x{max(heights)})")
        print(f"     Resize may lose detail, but necessary for memory constraints")
    
    imbalance = max(q1, q2, q3, q4) / min(q1, q2, q3, q4)
    if imbalance > 3:
        print(f"   ⚠ VA distribution is imbalanced (ratio: {imbalance:.1f}:1)")
        print(f"     Consider weighted sampling during training")
    else:
        print(f"   ✓ VA distribution is reasonably balanced")
    
    # 8. Display sample resized images
    print(f"\n8. SAMPLE RESIZED IMAGES")
    print(f"   Generating grid of 12 sample images (original vs resized)...")
    
    # Select 12 random images
    import random
    sample_indices = random.sample(range(len(existing_images)), min(12, len(existing_images)))
    
    fig2 = plt.figure(figsize=(20, 10))
    fig2.suptitle('Sample Images: Original (top) vs Resized 128×128 (bottom)', fontsize=16, y=0.98)
    
    for idx, sample_idx in enumerate(sample_indices):
        img_path = existing_images[sample_idx][0]
        v = existing_images[sample_idx][1]
        a = existing_images[sample_idx][2]
        
        # Load original
        img_original = Image.open(img_path).convert('RGB')
        
        # Resize to 128x128 (same as training)
        img_resized = img_original.resize((128, 128), Image.BILINEAR)
        
        # Plot original
        ax_orig = plt.subplot(4, 6, idx + 1)
        ax_orig.imshow(img_original)
        ax_orig.set_title(f'{img_original.size[0]}×{img_original.size[1]}', fontsize=9)
        ax_orig.axis('off')
        
        # Plot resized
        ax_resize = plt.subplot(4, 6, idx + 13)
        ax_resize.imshow(img_resized)
        ax_resize.set_title(f'V={v:.1f}, A={a:.1f}', fontsize=9, color='blue')
        ax_resize.axis('off')
    
    plt.tight_layout()
    
    # Save to same directory as script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    samples_path = os.path.join(script_dir, 'dataset_samples.png')
    plt.savefig(samples_path, dpi=150, bbox_inches='tight')
    print(f"   Saved to: {samples_path}")
    
    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"  - dataset_analysis.png: Statistics and distributions")
    print(f"  - dataset_samples.png: Sample resized images")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    img_dir = r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Data\Landscape"
    label_csv = r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Data\EmotionLabel\all_photos_valence_arousal.csv"
    
    analyze_dataset(img_dir, label_csv)
