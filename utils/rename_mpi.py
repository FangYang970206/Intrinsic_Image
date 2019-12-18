import os

source_path = "F:\\sintel\\gray_shading"
target_path = "F:\\sintel\\shading"

folders = os.listdir(source_path)

for folder in folders:
    source_names = os.listdir(os.path.join(source_path, folder))
    target_names = os.listdir(os.path.join(target_path, folder))
    for i in range(len(source_names)):
        os.rename(os.path.join(source_path, folder, source_names[i]), os.path.join(source_path, folder, target_names[i]))