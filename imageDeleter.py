import os

folder = "Input/Z"  # Folder path

# Get a list of image file names in the folder
image_files = [file for file in os.listdir(folder) if file.endswith(".jpg")]

# Sort the image file names in ascending order
image_files.sort()

# Iterate over the image file names starting from the second image
for i in range(1, len(image_files), 3):
    image_to_delete_1 = image_files[i]
    image_to_delete_2 = image_files[i+1]

    path_to_delete_1 = os.path.join(folder, image_to_delete_1)
    path_to_delete_2 = os.path.join(folder, image_to_delete_2)

    os.remove(path_to_delete_1)
    os.remove(path_to_delete_2)

    print(f"Deleted {image_to_delete_1}")
    print(f"Deleted {image_to_delete_2}")
