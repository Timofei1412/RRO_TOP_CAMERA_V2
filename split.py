import cv2
import numpy as np
import argparse
import sys
import os
from typing import Tuple, Optional, List


# ============================================================
# 1. Нарезка поля на секции
# ============================================================
def slice_field(image: np.ndarray,
                grid: Tuple[int, int] = (8, 8)) -> List[Tuple[int, int, int, np.ndarray]]:
    """
    Разрезает изображение поля на grid[1] × grid[0] секций.
    Возвращает список кортежей: [(index, row, col, section_bgr), ...]
    """
    H, W = image.shape[:2]
    rows, cols = grid
    cell_w = W / cols
    cell_h = H / rows
    sections = []
    for r in range(rows):
        for c in range(cols):
            x1 = int(round(c * cell_w))
            y1 = int(round(r * cell_h))
            x2 = int(round((c + 1) * cell_w))
            y2 = int(round((r + 1) * cell_h))
            x2 = min(x2, W)
            y2 = min(y2, H)
            section = image[y1:y2, x1:x2]
            index = r * cols + c
            sections.append((index, r, c, section))
    return sections


# ============================================================
# 2. Сохранение секций на диск
# ============================================================
def save_sections(sections: List[Tuple[int, int, int, np.ndarray]],
                  out_dir: str,
                  prefix: str = "section",
                  ext: str = "png") -> None:
    os.makedirs(out_dir, exist_ok=True)
    max_index = max(s[0] for s in sections)
    num_digits = len(str(max_index))
    for index, row, col, section_img in sections:
        fname = f"{prefix}_{str(index).zfill(num_digits)}.{ext}"
        fpath = os.path.join(out_dir, fname)
        cv2.imwrite(fpath, section_img)


# ============================================================
# 3. Отладочная визуализация сетки
# ============================================================
def show_grid(image: np.ndarray,
              grid: Tuple[int, int] = (8, 8),
              max_dim: int = 1000) -> None:
    H, W = image.shape[:2]
    rows, cols = grid
    cell_w = W / cols
    cell_h = H / rows
    dbg = image.copy()
    for c in range(1, cols):
        x = int(round(c * cell_w))
        cv2.line(dbg, (x, 0), (x, H), (0, 255, 0), 2)
    for r in range(1, rows):
        y = int(round(r * cell_h))
        cv2.line(dbg, (0, y), (W, y), (0, 255, 0), 2)
    for r in range(rows):
        for c in range(cols):
            index = r * cols + c
            cx = int((c + 0.5) * cell_w)
            cy = int((r + 0.5) * cell_h)
            cv2.rectangle(dbg, (cx - 30, cy - 20), (cx + 30, cy + 15), (0, 0, 0), -1)
            cv2.putText(dbg, str(index), (cx - 25, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.rectangle(dbg, (0, 0), (W - 1, H - 1), (0, 0, 255), 4)
    h_disp, w_disp = dbg.shape[:2]
    if max(h_disp, w_disp) > max_dim:
        scale = max_dim / float(max(h_disp, w_disp))
        dbg = cv2.resize(dbg, (int(w_disp * scale), int(h_disp * scale)),
                         interpolation=cv2.INTER_AREA)
    cv2.imshow("Field Grid (8x8)", dbg)
    print("Нажмите любую клавишу в окне OpenCV для закрытия...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# 4. Главная функция модуля
# ============================================================
def run(normalized_field: np.ndarray,
        out_dir: str = "output",
        grid: Tuple[int, int] = (8, 8),
        prefix: str = "section",
       debug: bool = False) -> List[Tuple[int, int, int, np.ndarray]]:
    if debug:
        show_grid(normalized_field, grid=grid)
    sections = slice_field(normalized_field, grid=grid)
    sections_dir = os.path.join(out_dir, "sections")
    save_sections(sections, sections_dir, prefix=prefix)
    return sections


# ============================================================
# 5. CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Нарезка нормализованного поля RRO на секции 8×8"
    )
    ap.add_argument("input",
                    help="Путь к нормализованному изображению поля "
                         "(результат работы normalize.py, например field.png)")
    ap.add_argument("-d", "--debug", action="store_true",
                    help="Показать изображение с наложенной сеткой и номерами")
    args = ap.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        print(f"Не удалось прочитать файл: {args.input}")
        sys.exit(1)

    H, W = image.shape[:2]
    print(f"   Размер: {W}×{H}")

    # === Фиксированные значения (больше не настраиваются из CLI) ===
    sections = run(image,
                   out_dir="sections",
                   grid=(8, 8),
                   prefix="section",
                   debug=args.debug)



if __name__ == "__main__":
    main()