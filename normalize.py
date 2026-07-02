import cv2
import numpy as np
import argparse
import sys
import os
import json
from typing import Optional, Tuple, Dict

DEFAULT_CONFIG = {
    'hsv': {
        'sat_min': 133,
        'val_min': 42,
    },
    'morph': {
        'close_iter': 0,
        'open_iter': 0,
        'erode_iter': 2,
        'dilate_iter': 2,
    },
    'ransac': {
        'iterations': 600,
        'inlier_threshold': 7,
        'min_inlier_ratio': 0.05,
        'min_points_per_edge': 40,
    },
}
LAST_QUAD_PATH = "last_quad.json"

def load_last_quad() -> Optional[np.ndarray]:
    if os.path.exists(LAST_QUAD_PATH):
        try:
            with open(LAST_QUAD_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                quad = np.array(data['quad'], dtype=np.float32)
                if quad.shape == (4, 2):
                    return quad
        except Exception:
            pass
    return None

def save_last_quad(quad: np.ndarray):
    try:
        data = {'quad': quad.tolist()}
        with open(LAST_QUAD_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Warning: Не удалось сохранить координаты: {e}")

class State:
    def __init__(self):
        self.config = {k: v.copy() for k, v in DEFAULT_CONFIG.items()}

state = State()

def grab_frame(source: str) -> np.ndarray:
    if source.lower().startswith("rtsp://") or source.lower().startswith("rtmp://"):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось подключиться к потоку: {source}")
        for _ in range(5):
            cap.grab()
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError("Кадр из потока не получен")
        return frame
    else:
        frame = cv2.imread(source)
        if frame is None:
            raise RuntimeError(f"Не удалось прочитать файл: {source}")
        return frame


def get_red_mask(bgr: np.ndarray, config: Optional[dict] = None) -> np.ndarray:
    if config is None:
        config = state.config
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat_min = config['hsv']['sat_min']
    val_min = config['hsv']['val_min']

    lower1 = np.array([0, sat_min, val_min])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, sat_min, val_min])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    if config['morph']['close_iter'] > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=config['morph']['close_iter'])
    if config['morph']['open_iter'] > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern, iterations=config['morph']['open_iter'])
    if config['morph']['erode_iter'] > 0:
        mask = cv2.erode(mask, kern, iterations=config['morph']['erode_iter'])
    if config['morph']['dilate_iter'] > 0:
        mask = cv2.dilate(mask, kern, iterations=config['morph']['dilate_iter'])

    return mask


def ransac_fit_line(points: np.ndarray, iterations: int = 300, inlier_threshold: float = 8.0,
                    min_inlier_ratio: float = 0.25) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple]]:
    if points is None or len(points) < 2:
        return None
    n = len(points)
    min_inliers = max(2, int(n * min_inlier_ratio))

    best_count = 0
    best_inliers = None
    best_line = None

    rng = np.random.default_rng(42)

    for _ in range(iterations):
        idx = rng.choice(n, size=2, replace=False)
        p1, p2 = points[idx[0]], points[idx[1]]

        dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if dist < 20:
            continue

        a = p1[1] - p2[1]
        b = p2[0] - p1[0]
        c = p1[0] * p2[1] - p2[0] * p1[1]
        norm = np.hypot(a, b)
        if norm == 0:
            continue
        a, b, c = a / norm, b / norm, c / norm

        distances = np.abs(points[:, 0] * a + points[:, 1] * b + c)
        inliers = distances < inlier_threshold
        count = int(np.sum(inliers))

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_line = np.array([a, b, c])

    if best_count < min_inliers or best_inliers is None:
        return None

    inlier_points = points[best_inliers]
    if len(inlier_points) < 2:
        return None

    line = cv2.fitLine(inlier_points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = float(line[0][0]), float(line[1][0]), float(line[2][0]), float(line[3][0])
    a = -vy
    b = vx
    c = -(a * x0 + b * y0)
    norm = np.hypot(a, b)
    if norm > 1e-6:
        a, b, c = a / norm, b / norm, c / norm

    final_line = np.array([a, b, c])
    final_distances = np.abs(points[:, 0] * a + points[:, 1] * b + c)
    final_inliers = final_distances < inlier_threshold

    return final_line, final_inliers, (x0, y0, vx, vy)


def line_from_coeffs(coeffs: np.ndarray) -> Tuple[float, float, float, float]:
    a, b, c = coeffs
    vx, vy = -b, a
    if abs(b) > abs(a):
        x0 = 0.0
        y0 = -c / b
    else:
        x0 = -c / a
        y0 = 0.0
    return float(vx), float(vy), float(x0), float(y0)


def classify_points_to_edges(points: np.ndarray, H: int, W: int) -> Dict[str, np.ndarray]:
    if len(points) == 0:
        return {'top': np.array([]), 'right': np.array([]),
                'bottom': np.array([]), 'left': np.array([])}
    pts = points.astype(np.float64)
    x, y = pts[:, 0], pts[:, 1]
    d1 = y * W - x * H
    d2 = y * W + x * H - W * H

    return {
        'top': pts[(d1 < 0) & (d2 < 0)],
        'bottom': pts[(d1 > 0) & (d2 > 0)],
        'left': pts[(d1 < 0) & (d2 > 0)],
        'right': pts[(d1 > 0) & (d2 < 0)],
    }


def intersect_lines(coeffs1: np.ndarray, coeffs2: np.ndarray) -> Optional[Tuple[float, float]]:
    a1, b1, c1 = coeffs1
    a2, b2, c2 = coeffs2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    return float(x), float(y)


def find_quad_ransac(mask: np.ndarray, config: Optional[dict] = None, debug_img: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    if config is None:
        config = state.config
    H, W = mask.shape[:2]
    ransac_cfg = config.get('ransac', DEFAULT_CONFIG['ransac'])

    ys, xs = np.where(mask > 0)
    if len(xs) < 100:
        if debug_img is not None:
            cv2.putText(debug_img, "TOO FEW RED PIXELS", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
        return None

    points = np.column_stack([xs, ys]).astype(np.float32)
    edges = classify_points_to_edges(points, H, W)

    if debug_img is not None:
        colors = {'top': (255, 0, 0), 'right': (0, 255, 0),
                  'bottom': (0, 0, 255), 'left': (255, 255, 0)}
        for name, pts in edges.items():
            col = colors[name]
            for p in pts[::max(1, len(pts) // 200)]:
                cv2.circle(debug_img, (int(p[0]), int(p[1])), 2, col, -1)

    edge_lines = {}
    min_pts = ransac_cfg['min_points_per_edge']

    for name in ['top', 'bottom', 'left', 'right']:
        pts = edges[name]
        if len(pts) < min_pts:
            if debug_img is not None:
                cv2.putText(debug_img, f"{name}: {len(pts)} <min", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            continue

        result = ransac_fit_line(
            pts,
            iterations=ransac_cfg['iterations'],
            inlier_threshold=ransac_cfg['inlier_threshold'],
            min_inlier_ratio=ransac_cfg['min_inlier_ratio'],
        )

        if result is not None:
            coeffs, inliers, _ = result
            edge_lines[name] = coeffs

            if debug_img is not None:
                vx, vy, x0, y0 = line_from_coeffs(coeffs)
                ext = max(H, W) * 2
                pt1 = (int(x0 - vx * ext), int(y0 - vy * ext))
                pt2 = (int(x0 + vx * ext), int(y0 + vy * ext))
                colors_line = {'top': (255, 0, 255), 'bottom': (255, 0, 255),
                               'left': (0, 255, 255), 'right': (0, 255, 255)}
                cv2.line(debug_img, pt1, pt2, colors_line[name], 4)

                inlier_pts = pts[inliers]
                for p in inlier_pts[::max(1, len(inlier_pts) // 100)]:
                    cv2.circle(debug_img, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)

                n_inliers = int(np.sum(inliers))
                cv2.putText(debug_img, f"{name}: {n_inliers}/{len(pts)} inliers",
                            (20, 30 + 30 * ['top', 'bottom', 'left', 'right'].index(name)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if len(edge_lines) < 4:
        if debug_img is not None:
            cv2.putText(debug_img, f"RANSAC: only {len(edge_lines)}/4 edges",
                        (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        return None

    corners = []
    pairs = [
        ('top', 'left'),
        ('top', 'right'),
        ('bottom', 'right'),
        ('bottom', 'left'),
    ]
    for e1, e2 in pairs:
        pt = intersect_lines(edge_lines[e1], edge_lines[e2])
        if pt is None:
            return None
        corners.append(pt)

    quad = order_points(np.array(corners, dtype=np.float32))

    if not validate_quad(quad, mask.shape):
        if debug_img is not None:
            cv2.putText(debug_img, "RANSAC: validation failed", (50, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        return None

    if debug_img is not None:
        pts = quad.astype(int)
        cv2.polylines(debug_img, [pts], True, (0, 255, 0), 6)
        for i, p in enumerate(pts):
            cv2.circle(debug_img, tuple(p), 15, (255, 0, 255), -1)
            cv2.putText(debug_img, str(i), (p[0] + 20, p[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
        cv2.putText(debug_img, "RANSAC OK", (50, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

    return quad


def validate_quad(quad: Optional[np.ndarray], img_shape: Tuple[int, ...], min_area_ratio: float = 0.02,
                  max_area_ratio: float = 0.95, min_ratio: float = 0.3, max_ratio: float = 3.0) -> bool:
    if quad is None:
        return False
    H, W = img_shape[:2]
    img_area = H * W
    if not check_aspect_ratio(quad, min_ratio, max_ratio):
        return False
    try:
        area = abs(cv2.contourArea(quad.astype(np.float32)))
    except cv2.error:
        return False
    if area < min_area_ratio * img_area or area > max_area_ratio * img_area:
        return False
    for pt in quad:
        if pt[0] < -W * 0.1 or pt[0] > W * 1.1 or pt[1] < -H * 0.1 or pt[1] > H * 1.1:
            return False

    signs = []
    for i in range(4):
        val = (quad[(i + 1) % 4][0] - quad[i][0]) * (quad[(i + 2) % 4][1] - quad[i][1]) - \
              (quad[(i + 1) % 4][1] - quad[i][1]) * (quad[(i + 2) % 4][0] - quad[i][0])
        signs.append(val)

    if not all(s > 0 for s in signs) and not all(s < 0 for s in signs):
        return False
    return True


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def check_aspect_ratio(quad: np.ndarray, min_ratio: float = 0.5, max_ratio: float = 2.0) -> bool:
    w_top = np.linalg.norm(quad[0] - quad[1])
    w_bot = np.linalg.norm(quad[3] - quad[2])
    h_left = np.linalg.norm(quad[0] - quad[3])
    h_right = np.linalg.norm(quad[1] - quad[2])
    width = (w_top + w_bot) / 2.0
    height = (h_left + h_right) / 2.0
    if height == 0:
        return False
    return min_ratio <= (width / height) <= max_ratio


def warp_field(image: np.ndarray, quad: np.ndarray,
               out_w: int = 1000, out_h: int = 1000) -> np.ndarray:
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(image, M, (out_w, out_h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def crop_1_percent(img: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    h, w = img.shape[:2]
    dy, dx = int(h * 0.01), int(w * 0.01)
    cropped = img[dy:h - dy, dx:w - dx]
    if target_size is not None:
        cropped = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)
    return cropped


INSTRUCTIONS_HEIGHT = 80

def manual_quad_input(frame: np.ndarray, max_display_size: Tuple[int, int] = (1200, 800)) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    scale_w = max_display_size[0] / w
    scale_h = max_display_size[1] / h
    scale = min(scale_w, scale_h, 1.0)

    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        display_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        display_frame = frame.copy()
        new_w, new_h = w, h
        scale = 1.0

    points_display, points_original = [], []
    window_name = "Manual Corner Selection - Click 4 corners (TL, TR, BR, BL)"
    instructions = np.zeros((INSTRUCTIONS_HEIGHT, new_w, 3), dtype=np.uint8)
    cv2.putText(instructions,
                "Click 4 corners in order: TL, TR, BR, BL. ENTER when done, ESC to cancel.",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(instructions,
                f"Image scaled to {new_w}x{new_h} (original {w}x{h}).",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    display = np.vstack([instructions, display_frame])
    cv2.imshow(window_name, display)

    def mouse_callback(event, x, y, flags, param):
        nonlocal points_display, points_original, display_frame, display
        if event == cv2.EVENT_LBUTTONDOWN:
            img_y = y - INSTRUCTIONS_HEIGHT
            if img_y < 0:
                return
            if len(points_display) < 4:
                points_display.append((x, img_y))
                orig_x, orig_y = int(x / scale), int(img_y / scale)
                points_original.append((orig_x, orig_y))
                cv2.circle(display_frame, (x, img_y), 6, (0, 255, 255), -1)
                cv2.putText(display_frame, str(len(points_display)), (x + 10, img_y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                display = np.vstack([instructions, display_frame])
                cv2.imshow(window_name, display)

    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        key = cv2.waitKey(10) & 0xFF
        if key == 13:
            if len(points_display) == 4:
                break
        elif key == 27:
            cv2.destroyWindow(window_name)
            return None

    cv2.destroyWindow(window_name)
    return order_points(np.array(points_original, dtype=np.float32))


def extract_field(source: str, debug: bool = False, out_size: Tuple[int, int] = (1000, 1000), hands: bool = False) -> Optional[np.ndarray]:
    frame = grab_frame(source)
    mask = get_red_mask(frame, state.config)
    quad = None
    method_used = None
    dbg_ransac = None

    if debug:
        dbg_ransac = frame.copy()
    quad = find_quad_ransac(mask, config=state.config, debug_img=dbg_ransac)
    if quad is not None:
        method_used = "RANSAC"

    if quad is not None:
        save_last_quad(quad)
    else:
        if hands:
            print("\nRANSAC не сработал.")
            print("Переход в ручной режим (флаг --hands активен)...")
            quad = manual_quad_input(frame)
            if quad is not None:
                method_used = "MANUAL"
                save_last_quad(quad)
            else:
                if debug:
                    cv2.destroyAllWindows()
                return None
        else:
            print("\nRANSAC не сработал.")
            print(f"Попытка загрузить последние сохранённые координаты из {LAST_QUAD_PATH}...")
            quad = load_last_quad()
            if quad is not None:
                method_used = "LAST_SAVED"
                print("Использован сохранённый квад из предыдущего запуска.")
            else:
                print(f"Warning: Файл {LAST_QUAD_PATH} не найден или невалиден.")
                if debug:
                    cv2.destroyAllWindows()
                return None

    if debug:
        show_debug("1. Original Frame", frame)
        show_debug("2. Red Mask", mask)
        show_debug("3. RANSAC Debug", dbg_ransac)

        dbg = frame.copy()
        pts = quad.astype(int)
        cv2.polylines(dbg, [pts], isClosed=True, color=(0, 255, 0), thickness=6)
        for i, p in enumerate(pts):
            cv2.circle(dbg, tuple(p), 15, (255, 0, 0), -1)
            cv2.putText(dbg, str(i), (p[0] + 20, p[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
        cv2.putText(dbg, f"Method: {method_used}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        show_debug("4. Detected Quad", dbg)

    if not check_aspect_ratio(quad):
        print(f"Странное соотношение сторон: {quad.tolist()}")

    field = warp_field(frame, quad, out_size[0], out_size[1])
    field = crop_1_percent(field, out_size)

    if debug:
        show_debug("5. Field (Warped)", field)
        print("Нажмите любую клавишу в окне OpenCV для закрытия...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return field


def show_debug(name: str, img: np.ndarray, max_dim: int = 1000):
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow(name, img)


def main():
    ap = argparse.ArgumentParser(description="Вырезание поля RRO по красной рамке")
    ap.add_argument("source", help="Путь к файлу ИЛИ rtsp://... URL")
    ap.add_argument("-o", "--output", default="field.png")
    ap.add_argument("-d", "--debug", action="store_true")
    ap.add_argument("--hands", action="store_true",
                    help="Разрешить ручной выбор углов, если RANSAC не сработал")
    args = ap.parse_args()

    try:
        field = extract_field(args.source, debug=args.debug,
                              out_size=(1000, 1000), hands=args.hands)
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)

    if field is not None:
        cv2.imwrite(args.output, field)
        print(f"Поле сохранено: {args.output} (1000x1000)")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()