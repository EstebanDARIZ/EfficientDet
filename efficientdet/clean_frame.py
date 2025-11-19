import os
import cv2
import argparse
from typing import List


'''
Clean Frame Tool
Interactive tool to clean bounding box annotations in text files.
Usage:
    python clean_frame.py --dir path/to/dataset --target-class 4 --view --verbose
Options:
    --dir: Directory containing images and .txt files.
    --target-class: Class to inspect (default: 4).
    --view: Display annotated images.
    --verbose: Print detailed actions.      
'''

# =========================================
#               FILE UTILITIES
# =========================================

def read_lines(file_path: str) -> List[str]:
    with open(file_path, "r") as f:
        return f.readlines()


def write_lines(file_path: str, lines: List[str]):
    with open(file_path, "w") as f:
        f.writelines(lines)


def delete_line(file_path: str, index: int):
    lines = read_lines(file_path)
    new = [l for i, l in enumerate(lines) if i != index - 1]
    write_lines(file_path, new)
    return lines


def delete_all_but_first(file_path: str):
    lines = read_lines(file_path)
    if lines:
        write_lines(file_path, [lines[0]])
    return lines


def change_class_all(file_path: str, new_id: int):
    lines = read_lines(file_path)
    out = [lines[0]]

    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 6:
            parts[0] = str(new_id)
            out.append(" ".join(parts) + "\n")
        else:
            out.append(line)

    write_lines(file_path, out)
    return lines


def change_class_specific(file_path: str, new_id: int, index: int):
    lines = read_lines(file_path)
    out = []

    for i, line in enumerate(lines):
        if i == index - 1:
            parts = line.split()
            if len(parts) == 6:
                parts[0] = str(new_id)
                out.append(" ".join(parts) + "\n")
            else:
                out.append(line)
        else:
            out.append(line)

    write_lines(file_path, out)
    return lines


# =========================================
#           VISUALISATION
# =========================================

def draw_predictions(image, txt_path: str):
    for line in read_lines(txt_path):
        parts = line.strip().split()
        if len(parts) != 6:
            continue

        class_id, score, x_min, y_min, x_max, y_max = map(float, parts)
        x_min, y_min, x_max, y_max = map(int, (x_min, y_min, x_max, y_max))

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        label = f"{score:.2f} Class:{int(class_id)}"
        cv2.putText(image, label, (x_min, y_min - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return image


# =========================================
#               MAIN LOGIC
# =========================================

def process_dataset(folder: str, target_class: int, view: bool, verbose: bool, target_nb_lines: int):

    print("\n=== CLEAN LABEL TOOL ===\n")
    print("Shortcuts:")
    print("  * : next file")
    print("  d : delete file")
    print("  f : keep only first line")
    print("  c : change all classes → 1")
    print("  p : change all classes → 5")
    print("  a/z/e : delete line 1/2/3")
    print("  t/y/u : change class line 1/2/3 → 5")
    print("  g/h/j : change class line 1/2/3 → 3")
    print("  b/n/, : change class line 1/2/3 → 1")
    print("  < : rollback")
    print("  k : stop processing\n")

    for fname in os.listdir(folder):

        if not fname.endswith(".txt"):
            continue

        txt_path = os.path.join(folder, fname)
        img_path = os.path.join(folder, fname.replace(".txt", ".jpg"))

        if not os.path.exists(img_path):
            if verbose:
                print(f"[IGNORED] Missing image for {fname}")
            continue

        lines = read_lines(txt_path)

        # Only process files containing the target class
        if target_class != -1 and target_class is not None:
            if not any(line.startswith(str(target_class)) for line in lines):
                continue
        # Only process files with up to target_nb_lines lines
        if len(lines) == target_nb_lines -1:
            continue

        if verbose:
            print(f"[PROCESS] {fname}")

        # Loop until user action
        windows_initialized = False
        original = None 
        kill = False

        while True:

            # Show annotated original
            img = draw_predictions(cv2.imread(img_path), txt_path)
            if view:
                if windows_initialized :
                    cv2.namedWindow("IMAGE", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("IMAGE", 1500, 900) 
                    windows_initialized = True
                cv2.imshow("IMAGE", img)

            key = cv2.waitKey(0) & 0xFF
            
            if original is None:
                original = lines[:]  # save backup

            # Actions
            if key == ord('f'):
                delete_all_but_first(txt_path)
                if verbose:
                    print(f"[KEPT FIRST] {fname}")

            elif key == ord('c'):
                change_class_all(txt_path, 1)
                if verbose:
                    print(f"[CHANGED ALL TO 1] {fname}")

            elif key == ord('p'):
                change_class_all(txt_path, 5)
                if verbose:
                    print(f"[CHANGED ALL TO 5] {fname}")

            elif key == ord('d'):
                os.remove(txt_path)
                os.remove(img_path)
                if verbose:
                    print(f"[DELETED] {fname} + image")
                break  # move to next file

            elif key == ord('a'):
                delete_line(txt_path, 1)
                if verbose:
                    print(f"[DELETED LINE 1] {fname}")

            elif key == ord('z'):
                delete_line(txt_path, 2)
                if verbose:
                    print(f"[DELETED LINE 2] {fname}")

            elif key == ord('e'):
                delete_line(txt_path, 3)
                if verbose:
                    print(f"[DELETED LINE 3] {fname}")

            elif key == ord('t'):
                change_class_specific(txt_path, 5, 1)
                if verbose:
                    print(f"[CHANGED LINE 1 TO 5] {fname}")

            elif key == ord('y'):
                change_class_specific(txt_path, 5, 2)
                if verbose:
                    print(f"[CHANGED LINE 2 TO 5] {fname}")

            elif key == ord('u'):
                change_class_specific(txt_path, 5, 3)
                if verbose:
                    print(f"[CHANGED LINE 3 TO 5] {fname}") 
            
            elif key == ord('g'):
                change_class_specific(txt_path, 3, 1)
                if verbose:
                    print(f"[CHANGED LINE 1 TO 3] {fname}")
            
            elif key == ord('h'):
                change_class_specific(txt_path, 3, 2)
                if verbose:
                    print(f"[CHANGED LINE 2 TO 3] {fname}")
            
            elif key == ord('j'):
                change_class_specific(txt_path, 3, 3)
                if verbose:
                    print(f"[CHANGED LINE 3 TO 3] {fname}")
            
            elif key == ord('b'):
                change_class_specific(txt_path, 1, 1)
                if verbose:
                    print(f"[CHANGED LINE 1 TO 1] {fname}")
            
            elif key == ord('n'):
                change_class_specific(txt_path, 1, 2)
                if verbose:
                    print(f"[CHANGED LINE 2 TO 1] {fname}")
            
            elif key == ord(','):
                change_class_specific(txt_path, 1, 3)
                if verbose:
                    print(f"[CHANGED LINE 3 TO 1] {fname}")

            elif key == ord('<'):
                write_lines(txt_path, original)
                if verbose:
                    print(f"[ROLLBACK] {fname}")    
                continue  # stay on the same file

            elif key == ord('*'):
                break  # move to next file  

            elif key == ord('k'):
                print("Stopping processing.")
                kill = True
                break

            #refresh lines after action
            lines = read_lines(txt_path)

        cv2.destroyAllWindows()
        if kill:
            break


# =========================================
#                 CLI
# =========================================

def cli():
    parser = argparse.ArgumentParser(
        description="Outil de nettoyage interactif des bounding boxes."
    )

    parser.add_argument(
        "--dir", required=True,
        help="Répertoire contenant les images et fichiers .txt"
    )

    parser.add_argument(
        "--target-class", type=int, default=-1,
        help="Classe à inspecter. (Default: ALL classes)"
    )

    parser.add_argument(
        "--view", action="store_true",
        help="Affiche les images annotées"
    )

    parser.add_argument(
        "--verbose", action="store_true",
        help="Affiche les actions en détail"
    )

    parser.add_argument(
        "--target-nb-lines", type=int, default=2,
        help="Nombre de lignes à afficher (Default: 2)"
    )

    args = parser.parse_args()

    process_dataset(
        folder=args.dir,
        target_class=args.target_class,
        view=args.view,
        verbose=args.verbose,
        target_nb_lines=args.target_nb_lines
    )


if __name__ == "__main__":
    cli()
