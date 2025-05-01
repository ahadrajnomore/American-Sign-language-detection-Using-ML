import os

folder = "LandmarkDataFromFolder/D"  # Folder path

# Get a list of image file names in the folder
image_files = [file for file in os.listdir(folder) if file.endswith(".jpg")]

# Sort the image file names in ascending order
image_files.sort()

# Rename the images sequentially
for i in range(len(image_files)):
    image_file = image_files[i]
    image_path = os.path.join(folder, image_file)

    new_image_name = "as" + str(i + 1) + ".jpg"
    new_image_path = os.path.join(folder, new_image_name)

    os.rename(image_path, new_image_path)
    print(f"Renamed {image_file} to {new_image_name}")
