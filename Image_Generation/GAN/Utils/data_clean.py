from PIL import Image
import pandas as pd
import os

# Read the image files in a folder
# Read the labels from a txt file
# The image file names are numbers, find the corresponding labels
# For example, the label file stores the valence values of all the images
# However, only a few images are used in the training set
# Find the corresponding labels for the training set images

# Read valence values from a txt file
valence_values = pd.read_csv(
    r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Data\EmotionLabel\valence_avg_10766_v2.txt",
    header=None,
)
arousal_values = pd.read_csv(
    r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Data\EmotionLabel\arousal_avg_10766_v2.txt",
    header=None,
)

# Read image file names from the folder
images = os.listdir(r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Data\All_photos")
# Extract only the numeric part of the image file names (assuming they are numbers)
images = [int(img.split(".")[0]) for img in images if img.split(".")[0].isdigit()]
# Sort the image file names
images.sort()


# Create lists for valence and arousal values for the images found in the folder
valence_list = [valence_values.iloc[image - 1, 0] for image in images]
arousal_list = [arousal_values.iloc[image - 1, 0] for image in images]

# Create DataFrame directly from the lists
combined_df = pd.DataFrame({
    'image': images,
    'valence': valence_list,
    'arousal': arousal_list
})

# Save to EmotionLabel folder - master CSV location
combined_df.to_csv(
    r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Data\EmotionLabel\all_photos_valence_arousal.csv",
    index=False,
    header=True,
)
