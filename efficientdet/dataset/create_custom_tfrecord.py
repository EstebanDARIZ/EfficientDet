import tensorflow as tf
import os
import glob
from PIL import Image
import io

# ➜ Adapte ici le nom des classes si tu veux
LABEL_MAP = {
    0: "Background",
    1: "Bait_calamar",
    2: "Bait_sardine",
    3: "Ray",
    4: "Sunfish",
    5: "Pilot_fish",
}

def create_tf_example(img_path, txt_path):
    # Lire l'image
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()

    img = Image.open(io.BytesIO(encoded_jpg))
    width, height = img.size
    filename = os.path.basename(img_path).encode("utf8")

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []

    # Lire le fichier labels
    with open(txt_path, "r") as f:
        for line in f:
            cls, x1, y1, x2, y2 = line.strip().split()
            cls = int(cls)

            # Normalisation
            x1 = float(x1) / width
            y1 = float(y1) / height
            x2 = float(x2) / width
            y2 = float(y2) / height

            xmins.append(x1)
            ymins.append(y1)
            xmaxs.append(x2)
            ymaxs.append(y2)

            classes.append(cls + 1)   # IMPORTANT : EfficientDet veut classes = 1..N
            classes_text.append(LABEL_MAP[cls].encode("utf8"))

    # Construction TFExample
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpg'])),

        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),

        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
    }))
    
    return tf_example


def create_tfrecord(images_dir, labels_dir, output_path):
    writer = tf.io.TFRecordWriter(output_path)
    imgs = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))

    for img_path in imgs:
        name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(labels_dir, name + ".txt")

        if not os.path.exists(txt_path):
            print(f"[WARNING] No label for {img_path}, skipping.")
            continue

        example = create_tf_example(img_path, txt_path)
        writer.write(example.SerializeToString())

    writer.close()
    print(f"[OK] TFRecord saved → {output_path}")


if __name__ == "__main__":
    create_tfrecord(
        images_dir="dataset/images",
        labels_dir="dataset/labels",
        output_path="tfrecord/mydata.tfrecord"
    )
