import cv2
import numpy as np
import argparse
import sys
import os
import json
from typing import Optional, Tuple, List, Dict, Any

DEFAULT_CONFIG = {
    'hough': {
        'threshold': 19,
        'minLineLength': 332,
        'maxLineGap': 66
    },
    'hsv': {
        'sat_min': 71,
        'val_min': 42
    },
    'morph': {
        'close_iter': 0,
        'open_iter': 0,
        'erode_iter': 2,
        'dilate_iter': 2
    },
    'filter': {
        'min_len_ratio': 0.1
    },
    'ransac': {
        'iterations': 600,
        'inlier_threshold': 7,
        'min_inlier_ratio': 0.05,
        'min_points_per_edge': 40
    }
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
        self.frame = None
        self.mask = None
        self.config = self._copy_config(DEFAULT_CONFIG)
        self.windows_created = False
        self.user_points = []
        self.refined_quad = None
        self.auto_quad = None
        self.saved_quad = None

    def _copy_config(self, config: dict) -> dict:
        return {
            'hough': config['hough'].copy(),
            'hsv': config['hsv'].copy(),
            'morph': config['morph'].copy(),
            'filter': config['filter'].copy(),
            'ransac': config['ransac'].copy(),
        }

    def update_config_from_trackbar(self):
        if not self.windows_created:
            return
        self.config['hough']['threshold'] = cv2.getTrackbarPos("Hough Thresh", "Debug Controls")
        self.config['hough']['minLineLength'] = cv2.getTrackbarPos("Min Line Len", "Debug Controls")
        self.config['hough']['maxLineGap'] = cv2.getTrackbarPos("Max Line Gap", "Debug Controls")
        self.config['hsv']['sat_min'] = cv2.getTrackbarPos("Sat Min", "Debug Controls")
        self.config['hsv']['val_min'] = cv2.getTrackbarPos("Val Min", "Debug Controls")
        self.config['morph']['close_iter'] = cv2.getTrackbarPos("Close Iter", "Debug Controls")
        self.config['morph']['open_iter'] = cv2.getTrackbarPos("Open Iter", "Debug Controls")
        self.config['morph']['erode_iter'] = cv2.getTrackbarPos("Erode Iter", "Debug Controls")
        self.config['morph']['dilate_iter'] = cv2.getTrackbarPos("Dilate Iter", "Debug Controls")
        self.config['filter']['min_len_ratio'] = cv2.getTrackbarPos("Min Len %", "Debug Controls") / 100.0
        self.config['ransac']['iterations'] = cv2.getTrackbarPos("RANSAC Iter", "Debug Controls")
        self.config['ransac']['inlier_threshold'] = cv2.getTrackbarPos("RANSAC Thresh", "Debug Controls")
        self.config['ransac']['min_inlier_ratio'] = cv2.getTrackbarPos("Min Inlier %", "Debug Controls") / 100.0

    def print_config_as_code(self):
        print("\n" + "=" * 70)
        print("ТЕКУЩИЕ ПАРАМЕТРЫ (скопируйте и вставьте в DEFAULT_CONFIG):")
        print("=" * 70)
        print("DEFAULT_CONFIG = {")
        for section, params in self.config.items():
            print(f"    '{section}': {{")
            for key, val in params.items():
                print(f"        '{key}': {val},")
            print("    },")
        print("}")
        print("=" * 70)


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


def ransac_fit_line(points: np.ndarray,
                    iterations: int = 300,
                    inlier_threshold: float = 8.0,
                    min_inlier_ratio: float = 0.25
                    ) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple]]:
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


def find_quad_ransac(mask: np.ndarray,
                     config: Optional[dict] = None,
                     debug_img: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
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


def find_quad_classic(mask: np.ndarray,
                      min_area_ratio: float = 0.02,
                      debug_img: Optional[np.ndarray] = None,
                      config: Optional[dict] = None) -> Optional[np.ndarray]:
    if config is None:
        config = state.config
    H, W = mask.shape[:2]
    lines = cv2.HoughLinesP(mask, rho=1, theta=np.pi / 180,
                            threshold=config['hough']['threshold'],
                            minLineLength=config['hough']['minLineLength'],
                            maxLineGap=config['hough']['maxLineGap'])
    quad_hough = None
    if lines is not None and len(lines) >= 4:
        quad_hough = find_quad_by_hough_enhanced(lines, H, W, debug_img, config['filter'])

    quad_hull = find_quad_by_convex_hull(mask, min_area_ratio)
    best_quad = select_best_quad(quad_hough, quad_hull, mask.shape, mask)

    if best_quad is None:
        if quad_hough is None and quad_hull is None:
            print("Warning: Оба алгоритма не нашли квад, fallback на minAreaRect...")
            return find_quad_by_min_area_rect(mask, min_area_ratio)
    return best_quad


def find_quad_by_hough_enhanced(lines: np.ndarray, H: int, W: int,
                                debug_img: Optional[np.ndarray] = None,
                                filter_params: Optional[dict] = None) -> Optional[np.ndarray]:
    if filter_params is None:
        filter_params = state.config['filter']
    segments = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        length = float(np.hypot(x2 - x1, y2 - y1))
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)
        segments.append((float(x1), float(y1), float(x2), float(y2), length, angle))

    min_len = max(H, W) * filter_params['min_len_ratio']
    segments = [s for s in segments if s[4] >= min_len]
    if len(segments) < 4:
        return None

    horizontals, verticals = [], []
    for s in segments:
        ang = s[5]
        if ang < 45 or ang > 135:
            horizontals.append(s)
        else:
            verticals.append(s)

    if len(horizontals) < 2 or len(verticals) < 2:
        return None

    h_lines = filter_external_lines(horizontals, H, W, is_horizontal=True,
                                    debug_img=debug_img, color=(255, 0, 0))
    v_lines = filter_external_lines(verticals, H, W, is_horizontal=False,
                                    debug_img=debug_img, color=(0, 0, 255))

    if h_lines is None or v_lines is None or len(h_lines) != 2 or len(v_lines) != 2:
        return None

    corners = []
    for hl in h_lines:
        for vl in v_lines:
            pt = line_intersection(hl, vl)
            if pt is None:
                return None
            corners.append(pt)

    if len(corners) != 4:
        return None
    quad = order_points(np.array(corners, dtype=np.float32))
    return quad


def filter_external_lines(segments: list, H: int, W: int,
                          is_horizontal: bool,
                          debug_img: Optional[np.ndarray] = None,
                          color: Tuple[int, int, int] = (255, 0, 0)) -> Optional[list]:
    if not segments or len(segments) < 2:
        return None

    if is_horizontal:
        sorted_segs = sorted(segments, key=lambda s: (s[1] + s[3]) / 2.0)
        mid_y = H / 2.0
        top_group = [s for s in sorted_segs if (s[1] + s[3]) / 2.0 < mid_y]
        bottom_group = [s for s in sorted_segs if (s[1] + s[3]) / 2.0 >= mid_y]
        if not top_group or not bottom_group:
            return None
        top_line = min(top_group, key=lambda s: (s[1] + s[3]) / 2.0)
        bottom_line = max(bottom_group, key=lambda s: (s[1] + s[3]) / 2.0)
        result_segments = [top_line, bottom_line]
    else:
        sorted_segs = sorted(segments, key=lambda s: (s[0] + s[2]) / 2.0)
        mid_x = W / 2.0
        left_group = [s for s in sorted_segs if (s[0] + s[2]) / 2.0 < mid_x]
        right_group = [s for s in sorted_segs if (s[0] + s[2]) / 2.0 >= mid_x]
        if not left_group or not right_group:
            return None
        left_line = min(left_group, key=lambda s: (s[0] + s[2]) / 2.0)
        right_line = max(right_group, key=lambda s: (s[0] + s[2]) / 2.0)
        result_segments = [left_line, right_line]

    lines = []
    for seg in result_segments:
        line = fit_line_to_segments([seg])
        lines.append(line)
        if debug_img is not None:
            x1, y1, x2, y2 = int(seg[0]), int(seg[1]), int(seg[2]), int(seg[3])
            cv2.line(debug_img, (x1, y1), (x2, y2), color, 2)
            vx, vy, x0, y0 = line
            extent = max(H, W) * 2
            pt1 = (int(x0 - vx * extent), int(y0 - vy * extent))
            pt2 = (int(x0 + vx * extent), int(y0 + vy * extent))
            cv2.line(debug_img, pt1, pt2, color, 4)
    return lines


def fit_line_to_segments(segments: list) -> Tuple[float, float, float, float]:
    if not segments:
        raise ValueError("Пустой список отрезков")
    points = []
    for s in segments:
        x1, y1, x2, y2, length, angle = s
        weight = max(1, int(length / 10.0))
        for _ in range(weight):
            points.append([[x1, y1]])
            points.append([[x2, y2]])

    if len(points) < 2:
        x1, y1, x2, y2, length, angle = segments[0]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dx, dy = (x2 - x1), (y2 - y1)
        n = float(np.hypot(dx, dy)) or 1.0
        return dx / n, dy / n, cx, cy

    pts_arr = np.array(points, dtype=np.float32)
    line = cv2.fitLine(pts_arr, cv2.DIST_L2, 0, 0.01, 0.01)
    return float(line[0][0]), float(line[1][0]), float(line[2][0]), float(line[3][0])


def line_intersection(line1, line2) -> Optional[Tuple[float, float]]:
    vx1, vy1, x1, y1 = line1
    vx2, vy2, x2, y2 = line2
    det = vx1 * vy2 - vy1 * vx2
    if abs(det) < 1e-6:
        return None
    t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / det
    return float(x1 + t * vx1), float(y1 + t * vy1)


def find_quad_by_min_area_rect(mask: np.ndarray,
                               min_area_ratio: float = 0.02) -> Optional[np.ndarray]:
    H, W = mask.shape[:2]
    img_area = H * W
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area_ratio * img_area:
        return None
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return order_points(np.int0(box).astype(np.float32))


def find_quad_by_convex_hull(mask: np.ndarray,
                             min_area_ratio: float = 0.02) -> Optional[np.ndarray]:
    H, W = mask.shape[:2]
    img_area = H * W
    points = cv2.findNonZero(mask)
    if points is None or len(points) < min_area_ratio * img_area:
        return None
    hull = cv2.convexHull(points)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    iterations = 0
    while len(approx) != 4 and iterations < 30:
        if len(approx) > 4:
            epsilon *= 1.15
        else:
            epsilon /= 1.15
        approx = cv2.approxPolyDP(hull, epsilon, True)
        iterations += 1

    if len(approx) == 4:
        return order_points(approx.reshape(4, 2).astype(np.float32))
    return find_quad_by_extreme_points(points)


def find_quad_by_extreme_points(points: np.ndarray) -> Optional[np.ndarray]:
    if points is None or len(points) < 4:
        return None
    pts = points.reshape(-1, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def validate_quad(quad: Optional[np.ndarray], img_shape: Tuple[int, ...],
                  min_area_ratio: float = 0.02, max_area_ratio: float = 0.95,
                  min_ratio: float = 0.3, max_ratio: float = 3.0) -> bool:
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


def score_quad_fit(quad: np.ndarray, mask: np.ndarray) -> float:
    H, W = mask.shape[:2]
    mask_quad = np.zeros((H, W), dtype=np.uint8)
    try:
        cv2.fillPoly(mask_quad, [quad.astype(np.int32)], 255)
    except cv2.error:
        return 0.0
    red_inside = cv2.countNonZero(cv2.bitwise_and(mask, mask_quad))
    red_total = cv2.countNonZero(mask)
    return red_inside / red_total if red_total > 0 else 0.0


def select_best_quad(quad1: Optional[np.ndarray], quad2: Optional[np.ndarray],
                     img_shape: Tuple[int, ...], mask: np.ndarray) -> Optional[np.ndarray]:
    valid1 = validate_quad(quad1, img_shape)
    valid2 = validate_quad(quad2, img_shape)
    if not valid1 and not valid2:
        return None
    if valid1 and not valid2:
        return quad1
    if valid2 and not valid1:
        return quad2
    score1 = score_quad_fit(quad1, mask) if quad1 is not None else 0.0
    score2 = score_quad_fit(quad2, mask) if quad2 is not None else 0.0
    return quad1 if score1 >= score2 else quad2


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


def check_aspect_ratio(quad: np.ndarray,
                       min_ratio: float = 0.5, max_ratio: float = 2.0) -> bool:
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


def manual_quad_input(frame: np.ndarray,
                      max_display_size: Tuple[int, int] = (1200, 800)) -> Optional[np.ndarray]:
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


def extract_field(source: str,
                  debug: bool = False,
                  out_size: Tuple[int, int] = (1000, 1000),
                  hands: bool = False) -> Optional[np.ndarray]:
    frame = grab_frame(source)
    mask = get_red_mask(frame, state.config)
    quad = None
    method_used = None
    dbg_ransac = None
    dbg_lines = None

    # Этап 1: RANSAC
    if debug:
        dbg_ransac = frame.copy()
    quad = find_quad_ransac(mask, config=state.config, debug_img=dbg_ransac)
    if quad is not None:
        method_used = "RANSAC (robust)"
    else:
        if debug:
            dbg_lines = frame.copy()
        quad = find_quad_classic(mask, debug_img=dbg_lines, config=state.config)
        if quad is not None:
            method_used = "Hough+ConvexHull (classic)"

    if quad is not None:
        save_last_quad(quad)
    else:
        if hands:
            print("\nВсе автоматические методы не сработали.")
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
            print("\nАвтоматические методы не сработали.")
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

    # Debug визуализация
    if debug:
        show_debug("1. Original Frame", frame)
        show_debug("2. Red Mask", mask)

        dbg = frame.copy()
        pts = quad.astype(int)
        cv2.polylines(dbg, [pts], isClosed=True, color=(0, 255, 0), thickness=6)
        for i, p in enumerate(pts):
            cv2.circle(dbg, tuple(p), 15, (255, 0, 0), -1)
            cv2.putText(dbg, str(i), (p[0] + 20, p[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
        cv2.putText(dbg, f"Method: {method_used}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        show_debug("3a. Detected Quad (green)", dbg)

        if dbg_ransac is not None:
            show_debug("3b. RANSAC Debug", dbg_ransac)
        if dbg_lines is not None:
            show_debug("3c. Hough Lines Debug", dbg_lines)

    if not check_aspect_ratio(quad):
        print(f"Странное соотношение сторон: {quad.tolist()}")

    field = warp_field(frame, quad, out_size[0], out_size[1])
    field = crop_1_percent(field, out_size)

    if debug:
        show_debug("4. Field (Warped)", field)
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


def on_trackbar_change(*args):
    state.update_config_from_trackbar()
    update_debug_windows()


def create_debug_gui(H: int, W: int):
    cv2.namedWindow("Debug Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Debug Controls", 900, 700)
    cv2.createTrackbar("Hough Thresh", "Debug Controls",
                       state.config['hough']['threshold'], 200, on_trackbar_change)
    cv2.createTrackbar("Min Line Len", "Debug Controls",
                       state.config['hough']['minLineLength'], 500, on_trackbar_change)
    cv2.createTrackbar("Max Line Gap", "Debug Controls",
                       state.config['hough']['maxLineGap'], 500, on_trackbar_change)
    cv2.createTrackbar("Sat Min", "Debug Controls",
                       state.config['hsv']['sat_min'], 255, on_trackbar_change)
    cv2.createTrackbar("Val Min", "Debug Controls",
                       state.config['hsv']['val_min'], 255, on_trackbar_change)
    cv2.createTrackbar("Close Iter", "Debug Controls",
                       state.config['morph']['close_iter'], 10, on_trackbar_change)
    cv2.createTrackbar("Open Iter", "Debug Controls",
                       state.config['morph']['open_iter'], 10, on_trackbar_change)
    cv2.createTrackbar("Erode Iter", "Debug Controls",
                       state.config['morph']['erode_iter'], 10, on_trackbar_change)
    cv2.createTrackbar("Dilate Iter", "Debug Controls",
                       state.config['morph']['dilate_iter'], 10, on_trackbar_change)
    cv2.createTrackbar("Min Len %", "Debug Controls",
                       int(state.config['filter']['min_len_ratio'] * 100), 50, on_trackbar_change)
    cv2.createTrackbar("RANSAC Iter", "Debug Controls",
                       state.config['ransac']['iterations'], 1000, on_trackbar_change)
    cv2.createTrackbar("RANSAC Thresh", "Debug Controls",
                       int(state.config['ransac']['inlier_threshold']), 30, on_trackbar_change)
    cv2.createTrackbar("Min Inlier %", "Debug Controls",
                       int(state.config['ransac']['min_inlier_ratio'] * 100), 80, on_trackbar_change)

    control_img = np.zeros((400, 900, 3), dtype=np.uint8)
    cv2.putText(control_img, "Debug Controls", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(control_img, "RANSAC algorithm active. Adjust trackbars to tune.",
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(control_img, "Press 'p' print, 'w' warp, 's' SAVE QUAD, 'c' clear, 'q' quit",
                (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
    cv2.putText(control_img, "Click 4 corners on 'Original Frame' for manual override",
                (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if state.saved_quad is not None:
        cv2.putText(control_img, "Loaded saved quad from last_quad.json (cyan)",
                    (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
    else:
        cv2.putText(control_img, "No saved quad found in last_quad.json",
                    (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Debug Controls", control_img)
    state.windows_created = True


def update_debug_windows():
    if not state.windows_created or state.frame is None:
        return
    H, W = state.frame.shape[:2]
    mask = get_red_mask(state.frame, state.config)
    state.mask = mask
    dbg_ransac = state.frame.copy()
    auto_quad = find_quad_ransac(mask, config=state.config, debug_img=dbg_ransac)
    method = "RANSAC"
    if auto_quad is None:
        auto_quad = find_quad_classic(mask, debug_img=None, config=state.config)
        method = "Classic"

    state.auto_quad = auto_quad
    img_orig = state.frame.copy()

    if state.saved_quad is not None:
        pts_saved = state.saved_quad.astype(int)
        cv2.polylines(img_orig, [pts_saved], True, (255, 200, 0), 2)
        cv2.putText(img_orig, "Saved (last_quad.json)", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)

    if auto_quad is not None:
        pts_auto = auto_quad.astype(int)
        cv2.polylines(img_orig, [pts_auto], True, (0, 255, 0), 2)
        cv2.putText(img_orig, f"Auto: {method}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    for i, pt in enumerate(state.user_points):
        cv2.circle(img_orig, pt, 8, (0, 255, 255), -1)
        cv2.putText(img_orig, str(i + 1), (pt[0] + 10, pt[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    show_debug("1. Original Frame", img_orig)
    show_debug("2. Red Mask", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    show_debug("3. RANSAC Debug (NEW)", dbg_ransac)

    dbg_hough = state.frame.copy()
    lines = cv2.HoughLinesP(mask, rho=1, theta=np.pi / 180,
                            threshold=state.config['hough']['threshold'],
                            minLineLength=state.config['hough']['minLineLength'],
                            maxLineGap=state.config['hough']['maxLineGap'])
    if lines is not None and len(lines) >= 4:
        quad_hough = find_quad_by_hough_enhanced(lines, H, W, dbg_hough,
                                                 state.config['filter'])
        status_hough = "VALID" if quad_hough is not None else "FAILED"
    else:
        status_hough = "NO LINES"
    cv2.putText(dbg_hough, f"Hough: {status_hough}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    show_debug("4. Algo 2: Hough (legacy)", dbg_hough)

    if state.refined_quad is not None:
        dbg_refined = state.frame.copy()
        pts_ref = state.refined_quad.astype(int)
        cv2.polylines(dbg_refined, [pts_ref], True, (255, 0, 255), 4)
        show_debug("5. Refined Quad (user)", dbg_refined)


def refine_quad_with_user_points(mask: np.ndarray, user_points: List[Tuple[int, int]],
                                 config: dict, max_dist: float = 30.0,
                                 max_angle_deg: float = 30.0) -> Optional[np.ndarray]:
    if len(user_points) != 4:
        return None
    H, W = mask.shape[:2]
    lines = cv2.HoughLinesP(mask, rho=1, theta=np.pi / 180,
                            threshold=config['hough']['threshold'],
                            minLineLength=config['hough']['minLineLength'],
                            maxLineGap=config['hough']['maxLineGap'])
    if lines is None or len(lines) < 4:
        return None

    segments = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        length = np.hypot(x2 - x1, y2 - y1)
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        segments.append((x1, y1, x2, y2, length, angle))

    refined_lines = []
    cos_max = np.cos(np.radians(max_angle_deg))
    for i in range(4):
        p1 = np.array(user_points[i], dtype=np.float32)
        p2 = np.array(user_points[(i + 1) % 4], dtype=np.float32)
        side_vec = p2 - p1
        side_len = np.linalg.norm(side_vec)
        if side_len < 1e-6:
            continue
        side_dir = side_vec / side_len
        good = []
        for seg in segments:
            dx, dy = seg[2] - seg[0], seg[3] - seg[1]
            seg_len = np.hypot(dx, dy)
            if seg_len < 1e-6:
                continue
            seg_dir = np.array([dx, dy]) / seg_len
            if abs(np.dot(side_dir, seg_dir)) < cos_max:
                continue
            seg_center = np.array([(seg[0] + seg[2]) / 2, (seg[1] + seg[3]) / 2])
            if abs(np.cross(side_dir, seg_center - p1)) > max_dist:
                continue
            good.append(seg)

        if len(good) < 2:
            pts_for_line = np.array([p1, p2], dtype=np.float32).reshape(-1, 2)
            line = cv2.fitLine(pts_for_line, cv2.DIST_L2, 0, 0.01, 0.01)
        else:
            all_points = []
            for seg in good:
                weight = max(1, int(seg[4] / 10.0))
                for _ in range(weight):
                    all_points.append([seg[0], seg[1]])
                    all_points.append([seg[2], seg[3]])
            line = cv2.fitLine(np.array(all_points, dtype=np.float32),
                               cv2.DIST_L2, 0, 0.01, 0.01)
        refined_lines.append(line)

    if len(refined_lines) != 4:
        return None

    corners = []
    for i in range(4):
        pt = line_intersection(refined_lines[i], refined_lines[(i + 1) % 4])
        if pt is None:
            return None
        corners.append(pt)
    return order_points(np.array(corners, dtype=np.float32))


def interactive_gui_mode(source: str):
    print("=" * 70)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ С GUI (RANSAC-powered)")
    print("=" * 70)
    print("\nНОВЫЙ АЛГОРИТМ: RANSAC — работает при 50-70% перекрытиях рамки")
    print("Трекбары: HSV, морфология, RANSAC (итерации, порог, % inliers)")
    print("Клавиши: 'p' = print config, 'w' = warp, 's' = save quad, 'c' = clear points, 'q' = quit\n")

    state.saved_quad = load_last_quad()
    if state.saved_quad is not None:
        print(f"Загружен сохранённый квад из {LAST_QUAD_PATH}")
    else:
        print(f"Файл {LAST_QUAD_PATH} не найден — сохранённого квада нет.")

    state.frame = grab_frame(source)
    H, W = state.frame.shape[:2]
    create_debug_gui(H, W)
    update_debug_windows()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(state.user_points) < 4:
                state.user_points.append((x, y))
                print(f"Точка {len(state.user_points)}: ({x}, {y})")
                if len(state.user_points) == 4:
                    refined = refine_quad_with_user_points(
                        state.mask, state.user_points, state.config)
                    if refined is not None:
                        state.refined_quad = refined
                        print("Уточнённый квад построен.")
                    else:
                        state.refined_quad = order_points(
                            np.array(state.user_points, dtype=np.float32))
                        print("Warning: Уточнение не удалось, использую клики.")
                    update_debug_windows()
            else:
                print("Уже есть 4 точки. 'c' — очистить.")
            update_debug_windows()

    cv2.setMouseCallback("1. Original Frame", mouse_callback)

    if state.auto_quad is None:
        print("\nWarning: АВТО-КВАД НЕ НАЙДЕН. Кликните 4 угла вручную.\n")
    else:
        print("\nКвад найден. При желании уточните кликами и нажмите 'w'.\n")

    while True:
        key = cv2.waitKey(100) & 0xFF
        state.update_config_from_trackbar()
        if key == ord('p'):
            state.print_config_as_code()
        if key == ord('q'):
            break
        if key == ord('c'):
            state.user_points.clear()
            state.refined_quad = None
            update_debug_windows()
        if key == ord('s'):
            if state.refined_quad is not None:
                save_last_quad(state.refined_quad)
                state.saved_quad = state.refined_quad
            elif state.auto_quad is not None:
                save_last_quad(state.auto_quad)
                state.saved_quad = state.auto_quad
            else:
                print("Warning: Нет квада для сохранения. Выберите 4 точки или настройте RANSAC.")
            update_debug_windows()
        if key == ord('w'):
            quad_to_use = None
            if state.refined_quad is not None:
                quad_to_use = state.refined_quad
                print("Использую уточнённый/ручной квад.")
            elif state.auto_quad is not None:
                quad_to_use = state.auto_quad
                print("Использую авто-квад.")
            elif state.saved_quad is not None:
                quad_to_use = state.saved_quad
                print("Использую сохранённый квад.")
            else:
                print("Нет квада. Задайте 4 точки или настройте RANSAC.")
                continue
            warped = warp_field(state.frame, quad_to_use, 1000, 1000)
            warped = crop_1_percent(warped, (1000, 1000))
            filename = "refined_field.png" if state.refined_quad else "auto_field.png"
            cv2.imwrite(filename, warped)
            cv2.imshow("Warped Field", warped)

    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Вырезание поля RRO по красной рамке")
    ap.add_argument("source", help="Путь к файлу ИЛИ rtsp://... URL")
    ap.add_argument("-o", "--output", default="field.png")
    ap.add_argument("-d", "--debug", action="store_true")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--hands", action="store_true",
                    help="Разрешить ручной выбор углов, если авто-методы не сработали")
    args = ap.parse_args()

    if args.gui:
        try:
            interactive_gui_mode(args.source)
        except KeyboardInterrupt:
            print("\nПрервано")
        finally:
            cv2.destroyAllWindows()
        return

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