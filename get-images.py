from datasets import load_dataset
import os

# Load the dataset
dataset = load_dataset("naver-clova-ix/cord-v2", split="train")

# Directory to save images
save_dir = "cord_images"
os.makedirs(save_dir, exist_ok=True)

# Save images
for i, row in enumerate(dataset):
    image = row["image"]  # Directly access the image object
    image_path = os.path.join(save_dir, f"image_{i}.png")
    
    # Save the image directly
    image.save(image_path)
    print(f"Saved {image_path}")

