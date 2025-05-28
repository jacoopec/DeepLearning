import os
from PIL import Image

# === Configuration ===
input_folder = "./"            # folder containing images
output_folder = "./output"   # folder to save resized images
target_size = (60, 30)              # (width, height)

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Supported image extensions
valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

# === Resize Images ===
for filename in os.listdir(input_folder):
    if filename.lower().endswith(valid_extensions):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("RGB")  # convert to RGB for consistency
        img_resized = img.resize(target_size)
        
        output_path = os.path.join(output_folder, filename)
        img_resized.save(output_path)

        print(f"Resized and saved: {filename}")

print("✅ Done resizing all images.")
