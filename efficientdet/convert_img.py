from PIL import Image
import os 

folder_path = "/home/esteban-dreau-darizcuren/doctorat/dataset/img_raw/total"

for j in os.listdir(folder_path):
    img_path = os.path.join(folder_path, j)
    save_folder_path = folder_path.replace("img_raw/total", "img_raw_rgb")
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    if j.endswith(".png"):
        img = Image.open(img_path).convert("RGB")
        save_img_path = os.path.join(save_folder_path, j.replace(".png", "_rgb.png"))
        img.save(save_img_path)
