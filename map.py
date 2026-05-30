import cv2
import numpy as np
import os
import json
import math
import argparse
import sys
from typing import List, Tuple, Dict, Optional

RED_LOWER1 = np.array([0, 70, 50])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([160, 70, 50])
RED_UPPER2 = np.array([180, 255, 255])

BLUE_LOWER = np.array([100, 70, 50])
BLUE_UPPER = np.array([130, 255, 255])

GREEN_LOWER = np.array([35, 50, 50])
GREEN_UPPER = np.array([85, 255, 255])

ORANGE_LOWER = np.array([5, 100, 100])
ORANGE_UPPER = np.array([25, 255, 255])

MIN_CONTOUR_AREA = 50
ASPECT_RATIO_THRESHOLD = 1.8
ROBOT_MIN_AREA_RATIO = 0.04

def get_masks(img: np.ndarray) -> Dict[str, np.ndarray]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_red1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    mask_red2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_blue = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    mask_green = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    mask_orange = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    kern_orange = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_OPEN, kern_orange)
    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_CLOSE, kern_orange)

    return {"red": mask_red, "blue": mask_blue, "green": mask_green, "orange": mask_orange}

def analyze_level(img: np.ndarray) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    white_pixels = cv2.countNonZero(thresh)
    total_pixels = img.shape[0] * img.shape[1]
    black_pixels = total_pixels - white_pixels
    return 1 if black_pixels > white_pixels else 0

def analyze_red_blue(mask_red: np.ndarray, mask_blue: np.ndarray,
                     row: int, col: int, img_shape: tuple) -> Dict:
    H, W = img_shape[:2]
    sec_area = H * W

    def process_mask(mask, is_red=True):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circles = []
        rectangles = []
        min_minor = min(H, W) * 0.12

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect
            if w == 0 or h == 0:
                continue
            if w < h:
                w, h = h, w
                angle += 90
            if h < min_minor:
                continue
            aspect_ratio = w / h
            center = (int(cx), int(cy))
            box = cv2.boxPoints(rect).astype(np.int32)
            x_min = int(np.min(box[:, 0]))
            x_max = int(np.max(box[:, 0]))
            y_min = int(np.min(box[:, 1]))
            y_max = int(np.max(box[:, 1]))
            orient = angle % 180
            if orient > 90:
                orient -= 180

            if aspect_ratio < ASPECT_RATIO_THRESHOLD:
                circles.append((center, box, area))
            else:
                if is_red:
                    edge_margin = min(H, W) * 0.05
                    is_border = False
                    if row == 0 and y_min < edge_margin: is_border = True
                    if row == 7 and y_max > H - edge_margin: is_border = True
                    if col == 0 and x_min < edge_margin: is_border = True
                    if col == 7 and x_max > W - edge_margin: is_border = True
                    if is_border:
                        continue
                if area < sec_area * 0.05 or area > sec_area * 0.40:
                    continue
                tube_type = 2 if abs(orient) < 45 else 1
                rectangles.append((center, tube_type, box, area))

        circles.sort(key=lambda x: x[2], reverse=True)
        rectangles.sort(key=lambda x: x[3], reverse=True)
        return circles, rectangles

    red_circles, red_rectangles = process_mask(mask_red, is_red=True)
    blue_circles, blue_rectangles = process_mask(mask_blue, is_red=False)

    ramp_type, ramp_angle, ramp_dir_precise = 0, 0.0, 0
    c_red_ramp, c_blue_ramp = None, None
    if red_circles and blue_circles:
        c_r, _, _ = red_circles[0]
        c_b, _, _ = blue_circles[0]
        
        dx = c_r[0] - c_b[0]
        dy = c_r[1] - c_b[1]
        angle_deg = math.degrees(math.atan2(dy, dx))
        
        orient = angle_deg % 180
        if orient > 90:
            orient -= 180
        ramp_type = 2 if abs(orient) < 45 else 1
        
        norm_angle = angle_deg
        if norm_angle < 0:
            norm_angle += 360
            
        if 315 <= norm_angle or norm_angle < 45:
            ramp_dir_precise = 'N'
        elif 45 <= norm_angle < 135:
            ramp_dir_precise = 'E'
        elif 135 <= norm_angle < 225:
            ramp_dir_precise = 'S'
        elif 225 <= norm_angle < 315:
            ramp_dir_precise = 'W'
            
        ramp_angle = angle_deg
        c_red_ramp, c_blue_ramp = c_r, c_b

    red_tube_type, red_tube_center, red_tube_box = 0, None, None
    if red_rectangles:
        red_tube_center, red_tube_type, red_tube_box, _ = red_rectangles[0]

    blue_tube_type, blue_tube_center, blue_tube_box = 0, None, None
    if blue_rectangles:
        blue_tube_center, blue_tube_type, blue_tube_box, _ = blue_rectangles[0]

    return {
        "ramp_type": ramp_type,
        "ramp_angle": round(ramp_angle, 2),
        "ramp_dir_precise": ramp_dir_precise,
        "c_red_ramp": c_red_ramp,
        "c_blue_ramp": c_blue_ramp,
        "red_tube_type": red_tube_type,
        "red_tube_center": red_tube_center,
        "red_tube_box": red_tube_box,
        "blue_tube_type": blue_tube_type,
        "blue_tube_center": blue_tube_center,
        "blue_tube_box": blue_tube_box,
    }

def analyze_green(mask_green: np.ndarray, row: int, col: int,
                  img_shape: tuple) -> Tuple[int, Optional[Tuple[int, int]], Optional[np.ndarray]]:
    H, W = img_shape[:2]
    contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
    if not valid_contours:
        return 0, None, None
    valid_contours.sort(key=cv2.contourArea, reverse=True)
    min_minor = min(H, W) * 0.12
    for cnt in valid_contours:
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        if w == 0 or h == 0:
            continue
        if w < h:
            w, h = h, w
            angle += 90
        if h < min_minor:
            continue
        box = cv2.boxPoints(rect).astype(np.int32)
        orient = angle % 180
        if orient > 90:
            orient -= 180
        center = (int(cx), int(cy))
        obj_type = 2 if abs(orient) < 45 else 1
        return obj_type, center, box
    return 0, None, None

def analyze_robot(mask_orange: np.ndarray, img_shape: tuple) -> Tuple[int, float]:
    H, W = img_shape[:2]
    sec_area = H * W
    contours, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0.0
    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    if area < sec_area * ROBOT_MIN_AREA_RATIO:
        return 0, area
    return 1, area

def analyze_section(img: np.ndarray, row: int, col: int) -> Dict:
    level = analyze_level(img)
    masks = get_masks(img)
    rb_data = analyze_red_blue(masks["red"], masks["blue"], row, col, img.shape)
    green_type, green_center, green_box = analyze_green(masks["green"], row, col, img.shape)
    robot_detected, robot_area = analyze_robot(masks["orange"], img.shape)

    if rb_data["ramp_type"] > 0:
        level = 0

    if robot_detected:
        if (rb_data["ramp_type"] > 0
                or rb_data["red_tube_type"] > 0
                or rb_data["blue_tube_type"] > 0
                or green_type > 0):
            robot_detected = 0
            robot_area = 0.0

    return {
        "level": level,
        "ramp": rb_data["ramp_type"],
        "ramp_angle": rb_data["ramp_angle"],
        "ramp_dir_precise": rb_data["ramp_dir_precise"],
        "ramp_centers": {"red": rb_data["c_red_ramp"], "blue": rb_data["c_blue_ramp"]},
        "green": green_type,
        "green_center": green_center,
        "green_box": green_box,
        "redTube": rb_data["red_tube_type"],
        "redTube_center": rb_data["red_tube_center"],
        "redTube_box": rb_data["red_tube_box"],
        "blueTube": rb_data["blue_tube_type"],
        "blueTube_center": rb_data["blue_tube_center"],
        "blueTube_box": rb_data["blue_tube_box"],
        "robot": robot_detected,
        "robot_area": robot_area,
    }

def fix_cut_objects(map_data: List[Dict], sec_w: int, sec_h: int):
    grid = {(item['row'], item['col']): item for item in map_data}
    obj_types = ['redTube', 'blueTube', 'green']
    thresh_x = sec_w * 0.25
    thresh_y = sec_h * 0.25
    merged_count = 0
    for r in range(8):
        for c in range(8):
            item = grid.get((r, c))
            if not item:
                continue
            if c < 7:
                right_item = grid.get((r, c + 1))
                if right_item:
                    for obj in obj_types:
                        if item[obj] > 0 and right_item[obj] > 0:
                            c1 = item.get(f"{obj}_center")
                            c2 = right_item.get(f"{obj}_center")
                            if c1 and c2:
                                if c1[0] > (sec_w - thresh_x) and c2[0] < thresh_x:
                                    if abs(c1[0] - sec_w) > abs(c2[0]):
                                        right_item[obj] = 0
                                    else:
                                        item[obj] = 0
                                    merged_count += 1
            if r < 7:
                bottom_item = grid.get((r + 1, c))
                if bottom_item:
                    for obj in obj_types:
                        if item[obj] > 0 and bottom_item[obj] > 0:
                            c1 = item.get(f"{obj}_center")
                            c2 = bottom_item.get(f"{obj}_center")
                            if c1 and c2:
                                if c1[1] > (sec_h - thresh_y) and c2[1] < thresh_y:
                                    if abs(c1[1] - sec_h) > abs(c2[1]):
                                        bottom_item[obj] = 0
                                    else:
                                        item[obj] = 0
                                    merged_count += 1

    if merged_count > 0:
        print(f"   Исправлено разрезанных объектов: {merged_count}")

def fix_robot_uniqueness(map_data: List[Dict]):
    robot_cells = [it for it in map_data if it.get("robot", 0) > 0]
    if len(robot_cells) <= 1:
        return
    best = max(robot_cells, key=lambda it: it.get("robot_area", 0.0))
    cleared = 0
    for it in robot_cells:
        if it is not best:
            it["robot"] = 0
            it["robot_area"] = 0.0
            cleared += 1
    if cleared > 0:
        print(f"   Робот продублирован в {cleared + 1} секциях — "
              f"оставлен в ({best['row']},{best['col']}) "
              f"(max area={best.get('robot_area', 0):.0f}px²)")

def draw_direction_arrow(canvas, center, obj_type, color, length=25):
    if not center or obj_type == 0:
        return
    cx, cy = center
    if obj_type == 2:
        cv2.arrowedLine(canvas, (cx, cy - length), (cx, cy + length), color, 2, tipLength=0.3)
    elif obj_type == 1:
        cv2.arrowedLine(canvas, (cx - length, cy), (cx + length, cy), color, 2, tipLength=0.3)

def draw_box(canvas, box, offset_x, offset_y, color, thickness=2):
    pts = box.copy()
    pts[:, 0] += offset_x
    pts[:, 1] += offset_y
    cv2.drawContours(canvas, [pts], 0, color, thickness)

def visualize_map_grid(map_data: List[Dict], sections: List[Tuple], max_dim: int = 1200):
    if not sections:
        return
    _, _, _, sample_img = sections[0]
    h, w = sample_img.shape[:2]
    rows, cols = 8, 8
    canvas = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
    data_by_idx = {item['index']: item for item in map_data}
    for index, r, c, img in sections:
        y1, y2 = r * h, (r + 1) * h
        x1, x2 = c * w, (c + 1) * w
        canvas[y1:y2, x1:x2] = img
        data = data_by_idx.get(index, {})
        level = data.get("level", 0)
        color_border = (0, 255, 255) if level == 1 else (100, 100, 100)
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), color_border, 2)

        ramp = data.get("ramp", 0)
        if ramp > 0:
            rc = data.get("ramp_centers", {})
            c_r, c_b = rc.get("red"), rc.get("blue")
            if c_r and c_b:
                pt1 = (x1 + c_r[0], y1 + c_r[1])
                pt2 = (x1 + c_b[0], y1 + c_b[1])
                cv2.line(canvas, pt1, pt2, (0, 255, 0), 2)
                cv2.circle(canvas, pt1, 6, (0, 0, 255), -1)
                cv2.circle(canvas, pt2, 6, (255, 0, 0), -1)
                cx_pt, cy_pt = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
                draw_direction_arrow(canvas, (cx_pt, cy_pt), ramp, (0, 165, 255), length=30)

        g_type = data.get("green", 0)
        g_c = data.get("green_center")
        g_box = data.get("green_box")
        if g_c and g_box is not None and g_type > 0:
            draw_box(canvas, g_box, x1, y1, (0, 255, 0), 2)
            draw_direction_arrow(canvas, (x1 + g_c[0], y1 + g_c[1]), g_type, (0, 255, 0), length=20)

        rt_type = data.get("redTube", 0)
        rt_c = data.get("redTube_center")
        rt_box = data.get("redTube_box")
        if rt_c and rt_box is not None and rt_type > 0:
            draw_box(canvas, rt_box, x1, y1, (0, 0, 255), 2)
            draw_direction_arrow(canvas, (x1 + rt_c[0], y1 + rt_c[1]), rt_type, (0, 0, 255), length=20)

        bt_type = data.get("blueTube", 0)
        bt_c = data.get("blueTube_center")
        bt_box = data.get("blueTube_box")
        if bt_c and bt_box is not None and bt_type > 0:
            draw_box(canvas, bt_box, x1, y1, (255, 0, 0), 2)
            draw_direction_arrow(canvas, (x1 + bt_c[0], y1 + bt_c[1]), bt_type, (255, 0, 0), length=20)

        robot = data.get("robot", 0)
        if robot > 0:
            cx = x1 + w // 2
            cy = y1 + h // 2
            cv2.circle(canvas, (cx, cy), min(w, h) // 4, (0, 140, 255), 4)
            cv2.putText(canvas, "ROBOT", (x1 + 5, y1 + h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

        cv2.putText(canvas, f"L:{level}", (x1 + 5, y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        info_y = y2 - 10
        if ramp > 0:
            ramp_precise = data.get("ramp_dir_precise", "")
            cv2.putText(canvas, f"R:{ramp}({ramp_precise})", (x1 + 5, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            info_y -= 12
        if g_type > 0:
            cv2.putText(canvas, f"G:{g_type}", (x1 + 5, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            info_y -= 12
        if rt_type > 0:
            cv2.putText(canvas, f"RT:{rt_type}", (x1 + 35, info_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        if bt_type > 0:
            cv2.putText(canvas, f"BT:{bt_type}", (x1 + 35, info_y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    h_can, w_can = canvas.shape[:2]
    if max(h_can, w_can) > max_dim:
        scale = max_dim / float(max(h_can, w_can))
        canvas = cv2.resize(canvas, (int(w_can * scale), int(h_can * scale)),
                            interpolation=cv2.INTER_AREA)
    cv2.imshow("Map Analysis (Grid)", canvas)

def visualize_map_schematic(clean_map_data: List[Dict],
                            cell_size: int = 100, max_dim: int = 1000):
    rows, cols = 8, 8
    margin = 40
    canvas_w = cols * cell_size + margin * 2
    canvas_h = rows * cell_size + margin * 2
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 240
    data_by_idx = {item['index']: item for item in clean_map_data}
    for r in range(rows):
        for c in range(cols):
            x1 = margin + c * cell_size
            y1 = margin + r * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            index = r * cols + c
            data = data_by_idx.get(index, {})
            level = data.get("level", 0)
            if level == 1:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 60, 60), -1)
            else:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (200, 200, 200), -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), 2)
            text_color = (255, 255, 255) if level == 1 else (0, 0, 0)
            cv2.putText(canvas, str(index), (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            cx = x1 + cell_size // 2
            cy = y1 + cell_size // 2
            cross_size = 4
            cv2.line(canvas, (cx - cross_size, cy), (cx + cross_size, cy), (80, 80, 80), 1)
            cv2.line(canvas, (cx, cy - cross_size), (cx, cy + cross_size), (80, 80, 80), 1)

            ramp = data.get("ramp", 0)
            if ramp > 0:
                arrow_len = 35
                if ramp == 2:
                    cv2.arrowedLine(canvas, (cx, cy - arrow_len), (cx, cy + arrow_len),
                                    (0, 165, 255), 4, tipLength=0.3)
                else:
                    cv2.arrowedLine(canvas, (cx - arrow_len, cy), (cx + arrow_len, cy),
                                    (0, 165, 255), 4, tipLength=0.3)
                ramp_precise = data.get("ramp_dir_precise", "")
                cv2.putText(canvas, f"R:{ramp}({ramp_precise})", (x1 + 5, y2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 2)

            g_type = data.get("green", 0)
            if g_type > 0:
                if g_type == 1:
                    rect_w, rect_h = 40, 14
                else:
                    rect_w, rect_h = 14, 40
                rx1 = cx - rect_w // 2
                ry1 = cy - rect_h // 2
                rx2 = cx + rect_w // 2
                ry2 = cy + rect_h // 2
                cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (0, 255, 0), 3)
                cv2.putText(canvas, "G", (x2 - 20, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 2)

            rt_type = data.get("redTube", 0)
            if rt_type > 0:
                if rt_type == 1:
                    rect_w, rect_h = 44, 12
                else:
                    rect_w, rect_h = 12, 44
                offset_x = -8 if rt_type == 2 else 0 
                offset_y = -8 if rt_type == 1 else 0
                rx1 = cx - rect_w // 2 + offset_x
                ry1 = cy - rect_h // 2 + offset_y
                rx2 = cx + rect_w // 2 + offset_x
                ry2 = cy + rect_h // 2 + offset_y
                cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (0, 0, 255), 3)
                cv2.putText(canvas, "RT", (x1 + 5, y2 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

            bt_type = data.get("blueTube", 0)
            if bt_type > 0:
                if bt_type == 1:
                    rect_w, rect_h = 44, 12
                else:
                    rect_w, rect_h = 12, 44
                offset_x = 8 if bt_type == 2 else 0
                offset_y = 8 if bt_type == 1 else 0
                rx1 = cx - rect_w // 2 + offset_x
                ry1 = cy - rect_h // 2 + offset_y
                rx2 = cx + rect_w // 2 + offset_x
                ry2 = cy + rect_h // 2 + offset_y
                cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (255, 0, 0), 3)
                cv2.putText(canvas, "BT", (x2 - 25, y2 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)

            robot = data.get("robot", 0)
            if robot > 0:
                L, W_tri = 30, 20
                pts = np.array([
                    [cx, cy - L],
                    [cx - W_tri, cy + L // 2],
                    [cx + W_tri, cy + L // 2],
                ], dtype=np.int32)
                cv2.fillPoly(canvas, [pts], (0, 140, 255))
                cv2.polylines(canvas, [pts], True, (0, 0, 0), 2)
                cv2.putText(canvas, "RBT", (x1 + 5, y1 + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 200), 2)

    cv2.putText(canvas, "Field Map Schematic (from JSON)", (margin, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    legend_y = canvas_h - 30
    cv2.putText(canvas, "Legend: R=Ramp, G=Green post, RT=Red tube, BT=Blue tube, RBT=Robot",
                (margin, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    h_can, w_can = canvas.shape[:2]
    if max(h_can, w_can) > max_dim:
        scale = max_dim / float(max(h_can, w_can))
        canvas = cv2.resize(canvas, (int(w_can * scale), int(h_can * scale)),
                            interpolation=cv2.INTER_AREA)
    cv2.imshow("Map Schematic", canvas)
    print("Press any key in any OpenCV window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def _json_serializer(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def run(sections: List[Tuple[int, int, int, np.ndarray]],
        out_dir: str,
        debug: bool = False) -> List[Dict]:
    map_data = []
    for index, row, col, img in sections:
        analysis = analyze_section(img, row, col)
        entry = {
            "index": index,
            "row": row,
            "col": col,
            "level": analysis["level"],
            "ramp": analysis["ramp"],
            "ramp_angle": analysis["ramp_angle"],
            "ramp_dir_precise": analysis["ramp_dir_precise"],
            "ramp_centers": analysis["ramp_centers"],
            "green": analysis["green"],
            "green_center": analysis["green_center"],
            "green_box": analysis["green_box"],
            "redTube": analysis["redTube"],
            "redTube_center": analysis["redTube_center"],
            "redTube_box": analysis["redTube_box"],
            "blueTube": analysis["blueTube"],
            "blueTube_center": analysis["blueTube_center"],
            "blueTube_box": analysis["blueTube_box"],
            "robot": analysis["robot"],
            "robot_area": analysis["robot_area"],
        }
        map_data.append(entry)
        
    _, _, _, sample_img = sections[0]
    sec_h, sec_w = sample_img.shape[:2]
    fix_cut_objects(map_data, sec_w, sec_h)
    fix_robot_uniqueness(map_data)

    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "map_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(map_data, f, indent=2, ensure_ascii=False, default=_json_serializer)

    clean_map_data = []
    for item in map_data:
        clean_map_data.append({
            "index": item["index"],
            "row": item["row"],
            "col": item["col"],
            "level": item["level"],
            "ramp": item["ramp"],
            "ramp_angle": item["ramp_angle"],
            "ramp_dir_precise": item["ramp_dir_precise"],
            "green": item["green"],
            "redTube": item["redTube"],
            "blueTube": item["blueTube"],
            "robot": item["robot"],
        })

    json_path = os.path.join(out_dir, "map.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_map_data, f, indent=4, ensure_ascii=False)

    if debug:
        for item in clean_map_data:
            parts = [f"Lvl:{item['level']}"]
            if item['ramp'] > 0: 
                parts.append(f"Ramp:{item['ramp']}({item['ramp_dir_precise']})")
            if item['green'] > 0: parts.append(f"Green:{item['green']}")
            if item['redTube'] > 0: parts.append(f"RT:{item['redTube']}")
            if item['blueTube'] > 0: parts.append(f"BT:{item['blueTube']}")
            if item['robot'] > 0: parts.append(f"ROBOT")
            print(f"   [{item['index']:2d}]   " + " | ".join(parts))
        visualize_map_grid(map_data, sections)
        visualize_map_schematic(clean_map_data)

    return clean_map_data

def main():
    ap = argparse.ArgumentParser(description="Analysis of RRO field sections (Map building)")
    ap.add_argument("--input-dir", default="output/sections")
    ap.add_argument("-o", "--out-dir", default="output")
    ap.add_argument("-d", "--debug", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Directory not found: {args.input_dir}")
        sys.exit(1)

    sections = []
    for i in range(64):
        fname = f"section_{i:02d}.png"
        path = os.path.join(args.input_dir, fname)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                sections.append((i, i // 8, i % 8, img))

    if not sections:
        print("No sections found.")
        sys.exit(1)

    print(f"   Loaded {len(sections)} sections.")
    run(sections, args.out_dir, debug=args.debug)

if __name__ == "__main__":
    main()