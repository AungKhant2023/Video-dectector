import os

def rename_images(folder_path, new_name_prefix, start_number=0):
    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter out .jpg, .jfif, and .jpeg files (case-insensitive)
    valid_extensions = ('.jpg', '.jfif', '.jpeg', '.png', '.webp')
    image_files = [f for f in files if f.lower().endswith(valid_extensions)]

    # Sort to maintain order
    image_files.sort()

    # Rename images starting from start_number
    for index, filename in enumerate(image_files):
        # Get full path
        old_path = os.path.join(folder_path, filename)

        # Extract original extension to keep it (jpg, jfif, jpeg)
        _, ext = os.path.splitext(filename)
        
        # New file name with starting offset
        new_name = f"{new_name_prefix}{start_number + index}{ext.lower()}"
        new_path = os.path.join(folder_path, new_name)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_name}")

    print("✅ All .jpg, .jpeg, and .jfif images renamed.")

# ------------------------------
# Example usage
# ------------------------------
folder = r"D:\image-testing\testing\anime\profile photo 1\facebook"
prefix = "Social - "

rename_images(folder, prefix)
