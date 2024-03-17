import os

def store_filename_without_extension(file_path):
    filename_with_extension = os.path.basename(file_path)
    filename_without_extension, _ = os.path.splitext(filename_with_extension)
    return filename_without_extension

file_path = "Images\Input\DemoImage.png"  # Change this to your file path
filename_without_extension = store_filename_without_extension(file_path)
print("Filename without extension:", filename_without_extension)