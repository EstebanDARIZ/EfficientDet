#!/usr/bin/env python3
"""
EfficientDet batch runner (uses ServingDriver, loads model once).

Usage example:
python inference_batch.py \
  --model efficientdet-d0 \
  --ckpt efficientdet-d0 \
  --input "/home/esteban-dreau-darizcuren/doctorat/dataset/img_raw/total/*.png" \
  --output output/ \
  --batch 4 \
  --min_score 0.3 \
  --display
  -- video path/to/video.mp4

If you're on a headless server, omit --display and use --output to save images.
"""
import os
import glob
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import time

# try import opencv, fallback to None (headless)
try:
    import cv2
except Exception:
    cv2 = None

# Ensure we import the repo modules (assumes script is run from repo root)
# from efficientdet package
from inference import ServingDriver
import hparams_config

# Optional: nice progress bar (if installed). If not, we quietly ignore.
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x


def enable_gpu_allow_growth_for_servingdriver():
    """
    Monkey-patch ServingDriver._build_session to enable GPU allow_growth.
    This helps avoid allocating all GPU memory at Session start.
    """
    import tensorflow.compat.v1 as tf

    orig = ServingDriver._build_session

    def _patched(self):
        sess_config = tf.ConfigProto()
        # allow GPU memory growth
        try:
            sess_config.gpu_options.allow_growth = True
        except Exception:
            pass
        if self.use_xla:
            sess_config.graph_options.optimizer_options.global_jit_level = (
                tf.OptimizerOptions.ON_2)
        return tf.Session(config=sess_config)

    ServingDriver._build_session = _patched

def resize_for_batch(imgs):
    """Resize all images of the batch to the same size (max H/W)."""
    import cv2
    h = max(im.shape[0] for im in imgs)
    w = max(im.shape[1] for im in imgs)

    resized = [cv2.resize(im, (w, h)) for im in imgs]
    return resized

import os

def save_predictions_and_image(image_path, img, threshold, detections, output_dir, min_score=0.1, new_class_name=None):
    """
    detections: [image_id, ymin, xmin, ymax, xmax, score, class]
    """
    base = os.path.basename(image_path)
    name = os.path.splitext(base)[0]
    txt_path = os.path.join(output_dir, f"{name}.txt")

    with open(txt_path, "w") as f:
        for det in detections:
            image_id, ymin, xmin, ymax, xmax, score, cls = det

            # filtrage des box invalides
            if score < min_score:
                continue
            if xmax <= xmin or ymax <= ymin:
                continue
            if score > threshold :
                f.write(f"{int(new_class_name)} {score:.4f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}\n")
    
    # chemin pour image annotée
    img_path = os.path.join(output_dir, f"{name}.jpg")
    Image.fromarray(img).save(img_path)

    print(f"Saved detections → {txt_path}")
    print(f"Saved raw image → {img_path}")



def load_image_to_uint8(path):
    img = Image.open(path).convert("RGB")
    return np.array(img).astype(np.uint8)

def process_video(args):
    # Vérifier sortie
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Activer allow_growth
    enable_gpu_allow_growth_for_servingdriver()

    print("Initializing EfficientDet for video...")
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
    print("Model loaded.")

    # Ouvrir vidéo
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open video: {args.video}")
        return

    frame_id = 0
    print("Processing video... (press S to save, Q to quit)")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # prédiction
        preds = driver.serve_images([frame_rgb])  # batch=1
        pred = preds[0]

        # Vérifier si une bbox dépasse min_score
        valid_dets = [d for d in pred if d[5] >= args.min_score]

        if len(valid_dets) > 0:
            # Annoter
            annotated = driver.visualize(
                frame_rgb,
                pred,
                min_score_thresh=args.min_score,
                max_boxes_to_draw=args.max_boxes,
                line_thickness=args.line_thickness
            )

            cv2.imshow("EfficientDet Video", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(0) & 0xFF

            if key == ord('c'):
                # sauvegarde frame + txt
                save_predictions_and_image(
                    image_path=f"frame_{frame_id}.jpg",
                    img=frame_rgb,
                    threshold=args.min_score,
                    detections=pred,
                    output_dir=args.output,
                    min_score=args.min_score,
                    new_class_name="1"  # Remplacer par le nom de classe souhaité
                )
                print(f"Frame {frame_id} saved.")
            
            if key == ord('s'):
                # sauvegarde frame + txt
                save_predictions_and_image(
                    image_path=f"frame_{frame_id}.jpg",
                    img=frame_rgb,
                    threshold=args.min_score,
                    detections=pred,
                    output_dir=args.output,
                    min_score=args.min_score,
                    new_class_name="2"  # Remplacer par le nom de classe souhaité
                )
                print(f"Frame {frame_id} saved.")
            
            if key == ord('r'):
                # sauvegarde frame + txt
                save_predictions_and_image(
                    image_path=f"frame_{frame_id}.jpg",
                    img=frame_rgb,
                    threshold=args.min_score,
                    detections=pred,
                    output_dir=args.output,
                    min_score=args.min_score,
                    new_class_name="3"  # Remplacer par le nom de classe souhaité
                )
                print(f"Frame {frame_id} saved.")

            if key == ord('l'):
                # sauvegarde frame + txt
                save_predictions_and_image(
                    image_path=f"frame_{frame_id}.jpg",
                    img=frame_rgb,
                    threshold=args.min_score,
                    detections=pred,
                    output_dir=args.output,
                    min_score=args.min_score,
                    new_class_name="4"  # Remplacer par le nom de classe souhaité
                )
                print(f"Frame {frame_id} saved.")

            elif key == ord('b'):
                print("Quit.")
                break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Video processing completed.")


def main(args):
    input_glob = args.input
    image_paths = sorted(glob.glob(input_glob))
    if not image_paths:
        print(f"No images found for pattern: {input_glob}")
        return

    out_dir = Path(args.output) if args.output else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Prevent TF from pre-allocating all memory
    enable_gpu_allow_growth_for_servingdriver()

    print("Initializing ServingDriver (this loads the TF graph + checkpoint) ...")
    driver = ServingDriver(
        model_name=args.model,
        ckpt_path=args.ckpt,
        batch_size=args.batch,
        min_score_thresh=args.min_score,
        max_boxes_to_draw=args.max_boxes,
        line_thickness=args.line_thickness,
        model_params=None
    )

    # build once
    driver.build()
    print("Model loaded.")

    # optional: fetch params for later use
    params = hparams_config.get_detection_config(args.model).as_dict()

    n = len(image_paths)
    batch = max(1, args.batch)
    idx = 0

    try:
        for start in tqdm(range(0, n, batch), desc="Batches"):
            batch_paths = image_paths[start:start + batch]
            imgs = [load_image_to_uint8(p) for p in batch_paths]

            # run inference (returns array per image)
            # preds = driver.serve_images(imgs)  # shape: [batch, num_dets, 7]
            imgs_resized = resize_for_batch(imgs)
            preds = driver.serve_images(imgs_resized)

            # visualize & save/display each
            for i, (img_np, pred, img_path) in enumerate(zip(imgs, preds, batch_paths)):
                annotated = driver.visualize(img_np, pred,
                                             min_score_thresh=args.min_score,
                                             max_boxes_to_draw=args.max_boxes,
                                             line_thickness=args.line_thickness)

                # annotated is numpy array (H,W,3) in uint8
                # Save to output if requested
                # if out_dir:
                #     out_path = out_dir / f"{start + i}.jpg"
                #     Image.fromarray(annotated).save(str(out_path))
                #     if not args.display:
                #         print(f"Wrote {out_path}")

                # Display if requested and cv2 available
                if args.display:
                    if cv2 is None:
                        # fallback: save temporary and print path
                        tmp_path = out_dir / f"{start + i}.jpg" if out_dir else Path(f"tmp_{start + i}.jpg")
                        Image.fromarray(annotated).save(str(tmp_path))
                        print(f"[no-opencv] saved preview to {tmp_path}")
                    else:
                        cv2.imshow("EfficientDet", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                        # waitKey: 0 = wait until key press, >0 = wait ms
                        key = cv2.waitKey(0) & 0xFF

                        # si 's' → Sauvegarder prédictions
                        if key == ord('s'):
                            save_predictions_and_image(img_path, imgs_resized[i], args.min_score,  pred, args.output)

                        # if 'q' pressed => quit
                        if key == ord('q'):
                            print("Quit key pressed. Exiting.")
                            raise KeyboardInterrupt

            # tiny pause to allow GPU to cool if requested
            if args.pause_between_batches > 0:
                time.sleep(args.pause_between_batches)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        if cv2 is not None:
            cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="EfficientDet batch runner (ServingDriver)")
    p.add_argument("--model", type=str, default="efficientdet-d0", help="model name")
    p.add_argument("--video", type=str, help="path to input video file")
    p.add_argument("--ckpt", type=str, default="efficientdet-d0", help="checkpoint path")
    p.add_argument("--input", type=str, required=True, help="glob pattern for input images, e.g. '/path/*.jpg'")
    p.add_argument("--output", type=str, default=None, help="output folder to save annotated images")
    p.add_argument("--batch", type=int, default=1, help="batch size for serve_images (best >1 if GPU available)")
    p.add_argument("--min_score", type=float, default=0.25, help="min score to visualize")
    p.add_argument("--max_boxes", type=int, default=200, help="max boxes to draw per image")
    p.add_argument("--line_thickness", type=int, default=2, help="box line thickness")
    p.add_argument("--display", action="store_true", help="display images with OpenCV")
    p.add_argument("--wait_key", type=int, default=0, help="cv2.waitKey ms (0 = wait keypress)")
    p.add_argument("--pause_between_batches", type=float, default=0.0, help="sleep seconds between batches (optional)")
    args = p.parse_args()


    if args.video:
        process_video(args)
    else:
        main(args)
