import cv2
import numpy as np
import argparse
import sys
import os
from typing import List, Tuple

def slice_field(image: np.ndarray, grid: Tuple[int, int] = (8, 8)) -> List[Tuple[int, int, int, np.ndarray]]:
    h, w = image.shape[:2]
    rows, cols = grid
    cell_w = w / cols
    cell_h = h / rows
    sections = []
    for r in range(rows):
        for c in range(cols):
            x1 = int(round(c * cell_w))
            y1 = int(round(r * cell_h))
            x2 = int(round((c + 1) * cell_w))
            y2 = int(round((r + 1) * cell_h))
            x2 = min(x2, w)
            y2 = min(y2, h)
            section = image[y1:y2, x1:x2]
            sections.append((r * cols + c, r, c, section))
    return sections

def save_sections(sections: List[Tuple[int, int, int, np.ndarray]], out_dir: str, prefix: str = "section", ext: str = "png") -> None:
    os.makedirs(out_dir, exist_ok=True)
    max_idx = max(s[0] for s in sections)
    digits = len(str(max_idx))
    for idx, _, _, img in sections:
        fname = f"{prefix}_{str(idx).zfill(digits)}.{ext}"
        cv2.imwrite(os.path.join(out_dir, fname), img)

def show_grid(image: np.ndarray, grid: Tuple[int, int] = (8, 8), max_dim: int = 1000) -> None:
    h, w = image.shape[:2]
    rows, cols = grid
    cell_w = w / cols
    cell_h = h / rows
    dbg = image.copy()

    for c in range(1, cols):
        x = int(round(c * cell_w))
        cv2.line(dbg, (x, 0), (x, h), (0, 255, 0), 2)
    for r in range(1, rows):
        y = int(round(r * cell_h))
        cv2.line(dbg, (0, y), (w, y), (0, 255, 0), 2)

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            cx = int((c + 0.5) * cell_w)
            cy = int((r + 0.5) * cell_h)
            cv2.rectangle(dbg, (cx - 30, cy - 20), (cx + 30, cy + 15), (0, 0, 0), -1)
            cv2.putText(dbg, str(idx), (cx - 25, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.rectangle(dbg, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)

    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        dbg = cv2.resize(dbg, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    cv2.imshow("Field Grid", dbg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run(normalized_field: np.ndarray, out_dir: str = "output", grid: Tuple[int, int] = (8, 8), prefix: str = "section", debug: bool = False) -> List[Tuple[int, int, int, np.ndarray]]:
    if debug:
        show_grid(normalized_field, grid)
    sections = slice_field(normalized_field, grid)
    save_sections(sections, os.path.join(out_dir, "sections"), prefix)
    return sections

def main():
    parser = argparse.ArgumentParser(description="Нарезка поля RRO на секции 8x8")
    parser.add_argument("input", help="Путь к изображению поля")
    parser.add_argument("-d", "--debug", action="store_true", help="Показать сетку")
    args = parser.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        print(f"Не удалось прочитать файл: {args.input}")
        sys.exit(1)

    print(f"Размер: {img.shape[1]}x{img.shape[0]}")
    run(img, out_dir="sections", grid=(8, 8), prefix="section", debug=args.debug)

if __name__ == "__main__":
    main()