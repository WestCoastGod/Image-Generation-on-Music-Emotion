import pandas as pd
import shutil
import os

# Read the master CSV
csv_path = r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Data\EmotionLabel\all_photos_valence_arousal.csv"
df = pd.read_csv(csv_path)

# Create bins for valence and arousal (low, mid, high)
df['v_bin'] = pd.cut(df['valence'], bins=[0, 3, 6, 9], labels=['low', 'mid', 'high'])
df['a_bin'] = pd.cut(df['arousal'], bins=[0, 3, 6, 9], labels=['low', 'mid', 'high'])

# Sample images from each VA combination (9 combinations)
# Get ~100 images per combination = ~4500 total images
selected = []
for v_bin in ['low', 'mid', 'high']:
    for a_bin in ['low', 'mid', 'high']:
        subset = df[(df['v_bin'] == v_bin) & (df['a_bin'] == a_bin)]
        if len(subset) >= 100:
            sampled = subset.sample(n=100, random_state=42)
        else:
            sampled = subset
        selected.append(sampled)

selected_df = pd.concat(selected, ignore_index=True)
print(f"Selected {len(selected_df)} images with good VA distribution")

# Create output folder
output_dir = r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Data\variate_images_for_test"
os.makedirs(output_dir, exist_ok=True)

# Copy images
source_dir = r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Data\All_photos"
copied = 0
for img_id in selected_df['image']:
    src = os.path.join(source_dir, f"{int(img_id)}.jpg")
    dst = os.path.join(output_dir, f"{int(img_id)}.jpg")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        copied += 1

print(f"Copied {copied} images to {output_dir}")
print(f"\nVA distribution:")
print(selected_df.groupby(['v_bin', 'a_bin']).size())
