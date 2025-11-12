from PIL import Image

img = Image.open("/home/esteban-dreau-darizcuren/doctorat/dataset/img_raw/Raie_1/4.png").convert("RGB")
img.save("/home/esteban-dreau-darizcuren/doctorat/dataset/img_raw_rgb/Raie_1/4_rgb.png")
