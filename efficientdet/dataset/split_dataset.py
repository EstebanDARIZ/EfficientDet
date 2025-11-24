import os
import shutil
import random
from glob import glob
import argparse


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_dataset(dataset_dir, train_ratio):
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")

    output_train_img = os.path.join(dataset_dir, "train/images")
    output_train_lbl = os.path.join(dataset_dir, "train/labels")
    output_val_img = os.path.join(dataset_dir, "val/images")
    output_val_lbl = os.path.join(dataset_dir, "val/labels")

    # Lire les images
    images = sorted(glob(os.path.join(images_dir, "*.jpg")))
    random.shuffle(images)

    total = len(images)
    train_count = int(total * train_ratio)

    train_list = images[:train_count]
    val_list = images[train_count:]

    print(f"Total images : {total}")
    print(f"Train : {len(train_list)}")
    print(f"Val   : {len(val_list)}")

    # Créer les dossiers
    for d in [output_train_img, output_train_lbl, output_val_img, output_val_lbl]:
        ensure_dir(d)

    # Copier les fichiers
    print("\n[INFO] Copie des images et labels...")

    for img_path in train_list:
        name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(labels_dir, name + ".txt")

        shutil.copy(img_path, output_train_img)
        shutil.copy(lbl_path, output_train_lbl)

    for img_path in val_list:
        name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(labels_dir, name + ".txt")

        shutil.copy(img_path, output_val_img)
        shutil.copy(lbl_path, output_val_lbl)

    print("\n[OK] Split terminé avec succès !")
    print(f"Dossier train : {os.path.join(dataset_dir, 'train')}")
    print(f"Dossier val   : {os.path.join(dataset_dir, 'val')}")


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train and val subsets.")
    parser.add_argument("--dataset", type=str, default="dataset", help="Chemin du dataset contenant images/ et labels/",)
    parser.add_argument("--ratio", type=float, default=0.8, help="Ratio du train (default: 0.8 → 80%% train / 20%% val)",)
    args = parser.parse_args()

    split_dataset(args.dataset, args.ratio)

if __name__ == "__main__":
    main()
