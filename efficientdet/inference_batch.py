#!/usr/bin/env python3
import os
import glob
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import time

try:
    import cv2
except Exception:
    cv2 = None

from inference import ServingDriver
import hparams_config

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x


# TF GPU memory control

def enable_gpu_allow_growth_for_servingdriver():
    import tensorflow.compat.v1 as tf
    orig = ServingDriver._build_session

    def _patched(self):
        cfg = tf.ConfigProto()
        try:
            cfg.gpu_options.allow_growth = True
        except:
            pass
        if self.use_xla:
            cfg.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_2
        return tf.Session(config=cfg)

    ServingDriver._build_session = _patched


# Tools

def resize_for_batch(imgs):
    h = max(im.shape[0] for im in imgs)
    w = max(im.shape[1] for im in imgs)
    return [cv2.resize(im, (w, h)) for im in imgs]


def load_image_to_uint8(path):
    return np.array(Image.open(path).convert("RGB")).astype(np.uint8)


def save_predictions_and_image(img_name, img, detections, out_dir, min_score, new_cls):
    """
    detections = [image_id, ymin, xmin, ymax, xmax, score, class]
    new_cls = string ("1","2","3","4")
    """
    name = os.path.splitext(img_name)[0]

    # TXT
    txt_path = os.path.join(out_dir, f"{name}.txt")
    with open(txt_path, "w") as f:
        for det in detections:
            _, ymin, xmin, ymax, xmax, score, cls = det
            if score >= min_score and xmax > xmin and ymax > ymin:
                f.write(f"{new_cls} {score:.4f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}\n")

    # JPG
    img_path = os.path.join(out_dir, f"{name}.jpg")
    Image.fromarray(img).save(img_path)

    print(f"Saved {txt_path}")
    print(f"Saved {img_path}")


# VIDEO processing

def process_video(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    enable_gpu_allow_growth_for_servingdriver()

    driver = ServingDriver(
        model_name=args.model,
        ckpt_path=args.ckpt,
        batch_size=1,
        min_score_thresh=args.min_score,
        max_boxes_to_draw=args.max_boxes,
        line_thickness=args.line_thickness,
        model_params=None
    )
    driver.build()
    print("Model loaded for video")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open video: {args.video}")
        return

    frame_id = 0
    print("Running video... (c/s/r/l/p to save with class 1/2/3/4/5, b to quit)")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pred = driver.serve_images([frame_rgb])[0]

        # keep only dets above threshold
        valid = [d for d in pred if d[5] >= args.min_score]

        if valid:
            print(f"Frame {frame_id}: {len(valid)} detections above threshold")
            annotated = driver.visualize(
                frame_rgb,
                pred,
                min_score_thresh=args.min_score,
                max_boxes_to_draw=args.max_boxes,
                line_thickness=args.line_thickness
            )

            cv2.imshow("EfficientDet Video", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(0) & 0xFF

            if key in [ord('c'), ord('s'), ord('r'), ord('l'), ord('p')]:
                cls_map = {ord('c'): "1", ord('s'): "2", ord('r'): "3", ord('l'): "4", ord('p'): "5"}
                cls_id = cls_map[key]
                print(f"Saving frame {frame_id} with class {cls_id}") 

                save_predictions_and_image(
                    img_name=f"frame_{frame_id}.jpg",
                    img=frame_rgb,
                    detections=pred,
                    out_dir=args.output,
                    min_score=args.min_score,
                    new_cls=cls_id
                )

            if key == ord('b'):
                print("Exit.")
                break
            print("Next")



        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
    print("✔ Video processing done.")


# IMAGE processing (batch)

def process_images(args):
    paths = sorted(glob.glob(args.input))
    if not paths:
        print(f"No images match: {args.input}")
        return

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    enable_gpu_allow_growth_for_servingdriver()

    driver = ServingDriver(
        model_name=args.model,
        ckpt_path=args.ckpt,
        batch_size=args.batch,
        min_score_thresh=args.min_score,
        max_boxes_to_draw=args.max_boxes,
        line_thickness=args.line_thickness,
        model_params=None
    )
    driver.build()
    print("✔ Model loaded for images")

    for start in tqdm(range(0, len(paths), args.batch), desc="Batches"):
        batch_paths = paths[start:start + args.batch]
        imgs = [load_image_to_uint8(p) for p in batch_paths]

        resized = resize_for_batch(imgs)
        preds = driver.serve_images(resized)

        for img_np, pred, img_path in zip(imgs, preds, batch_paths):
            annotated = driver.visualize(
                img_np,
                pred,
                min_score_thresh=args.min_score,
                max_boxes_to_draw=args.max_boxes,
                line_thickness=args.line_thickness
            )

            if args.display:
                cv2.imshow("EfficientDet", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(0) & 0xFF

                if key == ord('s'):
                    save_predictions_and_image(
                        img_name=os.path.basename(img_path),
                        img=img_np,
                        detections=pred,
                        out_dir=args.output,
                        min_score=args.min_score,
                        new_cls="1"
                    )

                if key == ord('q'):
                    print("Quit.")
                    return


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="efficientdet-d0")
    p.add_argument("--ckpt", type=str, default="efficientdet-d0")
    p.add_argument("--input", type=str, help="glob for images")
    p.add_argument("--video", type=str, help="path to video")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--min_score", type=float, default=0.25)
    p.add_argument("--max_boxes", type=int, default=200)
    p.add_argument("--line_thickness", type=int, default=2)
    p.add_argument("--display", action="store_true")
    args = p.parse_args()

    if args.video:
        process_video(args)
    elif args.input:
        process_images(args)
    else:
        print("You must provide either --input or --video")
