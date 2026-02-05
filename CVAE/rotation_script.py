from PIL import Image
import os

# input image path
img_path = "103072_class0_0.jpg"

# output folder
out_dir = "rotated"
os.makedirs(out_dir, exist_ok=True)

img = Image.open(img_path)

for angle in range(1, 360):
    rotated = img.rotate(angle, resample=Image.BICUBIC, expand=True)
    rotated.save(os.path.join(out_dir, f"rotss_{angle:03d}_class0_0.jpg"))

print("Done.")
