# This is for calcurating number of images in the folder
import os

def count_image_files(directory_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.bmp', '.tiff', '.webp'}

    image_count = 0
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_count += 1
    
    return image_count

# Change the path for each folder
directory_path = "./PlantVillage-Dataset/raw/color_copy/healthy"  

image_count = count_image_files(directory_path)
print(f"The number: {image_count}")
