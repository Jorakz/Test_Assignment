import os

# Define the translation dictionary
translate = {
    "cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly",
    "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel",
    "ragno": "spider"
}

# Define the path to the dataset
raw_img_path = "raw-img"

# Rename folders based on the translation dictionary
for folder in os.listdir(raw_img_path):
    folder_path = os.path.join(raw_img_path, folder)
    if os.path.isdir(folder_path) and folder in translate:
        new_folder_name = translate[folder]
        new_folder_path = os.path.join(raw_img_path, new_folder_name)
        os.rename(folder_path, new_folder_path)
        print(f"Renamed {folder} -> {new_folder_name}")

print("Folder renaming completed.")