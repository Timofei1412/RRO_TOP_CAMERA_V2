import cv2
import numpy as np
import os
import json
import math
import argparse
import sys
from typing import List, Tuple, Dict, Optional

RED_LOW1 = np.array([0, 70, 50])
RED_HIGH1 = np.array([10, 255, 255])
RED_LOW2 = np.array([160, 70, 50])
RED_HIGH2 = np.array([180, 255, 255])
BLUE_LOW = np.array([100, 70, 50])
BLUE_HIGH = np.array([130, 255, 255])
GREEN_LOW = np.array([35, 70, 70])
GREEN_HIGH = np.array([85, 255, 255])
ORANGE_LOW = np.array([5, 100, 100])
ORANGE_HIGH = np.array([25, 255, 255])

YELLOW_LOW = np.array([20, 100, 100])
YELLOW_HIGH = np.array([35, 255, 255])
CYAN_LOW = np.array([80, 70, 50])
CYAN_HIGH = np.array([100, 255, 255])
PURPLE_LOW = np.array([130, 50, 50])
PURPLE_HIGH = np.array([160, 255, 255])
WHITE_LOW = np.array([0, 0, 200])
WHITE_HIGH = np.array([180, 30, 255])
BLACK_LOW = np.array([0, 0, 0])
BLACK_HIGH = np.array([180, 255, 50])

MIN_AREA = 50
ASPECT_THRESH = 1.8
ROBOT_MIN_RATIO = 0.04

def get_masks(img: np.ndarray) -> Dict[str, np.ndarray]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, RED_LOW1, RED_HIGH1), cv2.inRange(hsv, RED_LOW2, RED_HIGH2))
    mask_blue = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)
    mask_green = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
    mask_orange = cv2.inRange(hsv, ORANGE_LOW, ORANGE_HIGH)

    kern_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kern_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kern_3)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kern_3)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kern_3)
    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_OPEN, kern_5)
    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_CLOSE, kern_5)

    return {"red": mask_red, "blue": mask_blue, "green": mask_green, "orange": mask_orange}

def analyze_level(img: np.ndarray) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
    white = cv2.countNonZero(thresh)
    total = img.shape[0] * img.shape[1]
    return 1 if (total - 2 * white) > 20 else 0

def analyze_red_blue(mask_red: np.ndarray, mask_blue: np.ndarray, row: int, col: int, img_shape: tuple) -> Dict:
    h, w = img_shape[:2]
    sec_area = h * w
    min_minor = min(h, w) * 0.12

    def process_mask(mask, is_red=True):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circles, rectangles = [], []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA: continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), angle = rect
            if rw == 0 or rh == 0: continue
            if rw < rh:
                rw, rh = rh, rw
                angle += 90
            if rh < min_minor: continue
            aspect = rw / rh
            center = (int(cx), int(cy))
            box = cv2.boxPoints(rect).astype(np.int32)
            x_min, x_max = int(np.min(box[:, 0])), int(np.max(box[:, 0]))
            y_min, y_max = int(np.min(box[:, 1])), int(np.max(box[:, 1]))
            orient = angle % 180
            if orient > 90: orient -= 180

            if aspect < ASPECT_THRESH:
                circles.append((center, box, area))
            else:
                if is_red:
                    margin = min(h, w) * 0.05
                    is_border = (row == 0 and y_min < margin) or \
                                (row == 7 and y_max > h - margin) or \
                                (col == 0 and x_min < margin) or \
                                (col == 7 and x_max > w - margin)
                    if is_border: continue
                if area < sec_area * 0.05 or area > sec_area * 0.40: continue
                tubes = 2 if abs(orient) < 45 else 1
                rectangles.append((center, tubes, box, area))
        circles.sort(key=lambda x: x[2], reverse=True)
        rectangles.sort(key=lambda x: x[3], reverse=True)
        return circles, rectangles

    red_c, red_r = process_mask(mask_red, True)
    blue_c, blue_r = process_mask(mask_blue, False)

    ramp_type, ramp_angle, ramp_dir = 0, 0.0, 0
    c_red_ramp, c_blue_ramp = None, None
    objData = 0
    if red_c and blue_c:
        cr, _, _ = red_c[0]
        cb, _, _ = blue_c[0]
        dx, dy = cr[0] - cb[0], cr[1] - cb[1]
        # print(row*8 + col, dx, dy)
        if ((abs(dx) - 50 > 0 and abs(dy) -  50 > 0)):
            #new fuiling station
            if (row not in [0, 7] and col not in [0, 7]):
                print(f"ITS a station {row * 8 + col}")
                objData = 1
        else:
            angle_deg = math.degrees(math.atan2(dy, dx))
            orient = angle_deg % 180
            if orient > 90: orient -= 180
            ramp_type = 2 if abs(orient) < 45 else 1

            norm = angle_deg
            if norm < 0: norm += 360
            if norm >= 315 or norm < 45: ramp_dir = 'N'
            elif 45 <= norm < 135: ramp_dir = 'E'
            elif 135 <= norm < 225: ramp_dir = 'S'
            else: ramp_dir = 'W'

            ramp_angle = angle_deg
            c_red_ramp, c_blue_ramp = cr, cb
            
    
    rt_type, rt_center, rt_box = 0, None, None
    if red_r: rt_center, rt_type, rt_box, _ = red_r[0]
    bt_type, bt_center, bt_box = 0, None, None
    if blue_r: bt_center, bt_type, bt_box, _ = blue_r[0]

    return {
        "ramp_type": ramp_type, "ramp_angle": round(ramp_angle, 2), "ramp_dir_precise": ramp_dir,
        "c_red_ramp": c_red_ramp, "c_blue_ramp": c_blue_ramp,
        "red_tube_type": rt_type, "red_tube_center": rt_center, "red_tube_box": rt_box,
        "blue_tube_type": bt_type, "blue_tube_center": bt_center, "blue_tube_box": bt_box,
        "object_data": objData,
    }

def analyze_green(mask_green: np.ndarray, row: int, col: int, img_shape: tuple) -> Tuple[int, Optional[Tuple[int, int]], Optional[np.ndarray]]:
    h, w = img_shape[:2]
    contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > MIN_AREA]
    if not valid: return 0, None, None
    valid.sort(key=cv2.contourArea, reverse=True)
    min_minor = min(h, w) * 0.12
    for cnt in valid:
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect
        if rw == 0 or rh == 0: continue
        if rw < rh: rw, rh = rh, rw; angle += 90
        if rh < min_minor: continue
        box = cv2.boxPoints(rect).astype(np.int32)
        orient = angle % 180
        if orient > 90: orient -= 180
        obj_type = 2 if abs(orient) < 45 else 1
        return obj_type, (int(cx), int(cy)), box
    return 0, None, None

def analyze_robot(mask_orange: np.ndarray, img: np.ndarray, img_shape: tuple, use_qr_flag: Optional[bool] = None) -> Tuple[int, float]:
    h, w = img_shape[:2]
    
    if use_qr_flag is None:
        contours, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: 
            return 0, 0.0
        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        if area < (h * w) * ROBOT_MIN_RATIO: 
            return 0, area
        return 1, area
    elif use_qr_flag is True:
        try:
            qr_decoder = cv2.QRCodeDetector()
            data, bbox, _ = qr_decoder.detectAndDecode(img)
            if data and data.strip() == "42" and bbox is not None:
                pts = bbox.reshape(4, 2).astype(np.int32)
                area = float(cv2.contourArea(pts))
                return 1, area
        except Exception:
            pass
        return 0, 0.0
    else: 
        try:
            # Новый API (OpenCV >= 4.7.0)
            # Используем DICT_4X4_100, так как ID 67 не помещается в DICT_4X4_50 (максимум ID = 49)
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
            parameters = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, _ = detector.detectMarkers(img)
        except AttributeError:
            # Старый API (OpenCV < 4.7.0)
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
            parameters = cv2.aruco.DetectorParameters_create()
            corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
            
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == 67:
                    pts = corners[i].reshape(4, 2).astype(np.int32)
                    area = float(cv2.contourArea(pts))
                    return 1, area
                    
        return 0, 0.0

def analyze_section(img: np.ndarray, row: int, col: int, use_qr_flag: Optional[bool] = None) -> Dict:
    level = analyze_level(img)
    masks = get_masks(img)
    rb = analyze_red_blue(masks["red"], masks["blue"], row, col, img.shape)
    g_type, g_center, g_box = analyze_green(masks["green"], row, col, img.shape)
    robot_det, robot_area = analyze_robot(masks["orange"], img, img.shape, use_qr_flag)

    if rb["ramp_type"] > 0: level = 0
    if robot_det and (rb["ramp_type"] > 0 or rb["red_tube_type"] > 0 or rb["blue_tube_type"] > 0 or g_type > 0):
        robot_det, robot_area = 0, 0.0

    return {
        "level": level, "ramp": rb["ramp_type"], "ramp_angle": rb["ramp_angle"],
        "ramp_dir_precise": rb["ramp_dir_precise"], "ramp_centers": {"red": rb["c_red_ramp"], "blue": rb["c_blue_ramp"]},
        "green": g_type, "green_center": g_center, "green_box": g_box,
        "redTube": rb["red_tube_type"], "redTube_center": rb["red_tube_center"], "redTube_box": rb["red_tube_box"],
        "blueTube": rb["blue_tube_type"], "blueTube_center": rb["blue_tube_center"], "blueTube_box": rb["blue_tube_box"],
        "robot": robot_det, "robot_area": robot_area,
        "obj": rb["object_data"], "obj_pos": 3 if rb["object_data"] != 0 else 0,
    }

def fix_cut_objects(map_data: List[Dict], sec_w: int, sec_h: int):
    grid = {(item['row'], item['col']): item for item in map_data}
    obj_types = ['redTube', 'blueTube', 'green']
    thresh_x, thresh_y = sec_w * 0.25, sec_h * 0.25
    merged = 0
    for r in range(8):
        for c in range(8):
            item = grid.get((r, c))
            if not item: continue
            if c < 7:
                right = grid.get((r, c + 1))
                if right:
                    for obj in obj_types:
                        if item[obj] > 0 and right[obj] > 0:
                            c1 = item.get(f"{obj}_center")
                            c2 = right.get(f"{obj}_center")
                            if c1 and c2 and c1[0] > (sec_w - thresh_x) and c2[0] < thresh_x:
                                if abs(c1[0] - sec_w) > abs(c2[0]): right[obj] = 0
                                else: item[obj] = 0
                                merged += 1
            if r < 7:
                bottom = grid.get((r + 1, c))
                if bottom:
                    for obj in obj_types:
                        if item[obj] > 0 and bottom[obj] > 0:
                            c1 = item.get(f"{obj}_center")
                            c2 = bottom.get(f"{obj}_center")
                            if c1 and c2 and c1[1] > (sec_h - thresh_y) and c2[1] < thresh_y:
                                if abs(c1[1] - sec_h) > abs(c2[1]): bottom[obj] = 0
                                else: item[obj] = 0
                                merged += 1
    if merged > 0: print(f"Исправлено разрезанных объектов: {merged}")

def fix_robot_uniqueness(map_data: List[Dict]):
    robots = [it for it in map_data if it.get("robot", 0) > 0]
    if len(robots) <= 1: return
    best = max(robots, key=lambda it: it.get("robot_area", 0.0))
    cleared = 0
    for it in robots:
        if it is not best:
            it["robot"] = 0
            it["robot_area"] = 0.0
            cleared += 1
    if cleared > 0:
        print(f"Робот продублирован в {cleared + 1} секциях — "
              f"оставлен в ({best['row']},{best['col']}) (area={best.get('robot_area', 0):.0f})")

def set_obj_cell(map_data: List[Dict], row: int, col: int, pos: int = 0) -> List[Dict]:
    for item in map_data:
        if item["row"] == row and item["col"] == col:
            item["obj"] = 1
            item["obj_pos"] = pos
            print(f"  [OBJ] Клетка ({row},{col}) помечена как waypoint (pos={pos})")
            return map_data
    print(f"  [WARN] Клетка ({row},{col}) не найдена в map_data")
    return map_data

def draw_arrow(canvas, center, obj_type, color, length=25):
    if not center or obj_type == 0: return
    cx, cy = center
    if obj_type == 2:
        cv2.arrowedLine(canvas, (cx, cy - length), (cx, cy + length), color, 2, tipLength=0.3)
    elif obj_type == 1:
        cv2.arrowedLine(canvas, (cx - length, cy), (cx + length, cy), color, 2, tipLength=0.3)

def draw_box(canvas, box, ox, oy, color, thickness=2):
    pts = box.copy()
    pts[:, 0] += ox
    pts[:, 1] += oy
    cv2.drawContours(canvas, [pts], 0, color, thickness)

def visualize_map_grid(map_data: List[Dict], sections: List[Tuple], max_dim: int = 1200):
    if not sections: return
    _, _, _, sample = sections[0]
    h, w = sample.shape[:2]
    canvas = np.zeros((h * 8, w * 8, 3), dtype=np.uint8)
    data_by_idx = {item['index']: item for item in map_data}

    for idx, r, c, img in sections:
        y1, y2, x1, x2 = r * h, (r + 1) * h, c * w, (c + 1) * w
        canvas[y1:y2, x1:x2] = img
        d = data_by_idx.get(idx, {})
        lvl = d.get("level", 0)
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255) if lvl == 1 else (100, 100, 100), 2)

        ramp = d.get("ramp", 0)
        if ramp > 0:
            rc = d.get("ramp_centers", {})
            cr, cb = rc.get("red"), rc.get("blue")
            if cr and cb:
                p1, p2 = (x1 + cr[0], y1 + cr[1]), (x1 + cb[0], y1 + cb[1])
                cv2.line(canvas, p1, p2, (0, 255, 0), 2)
                cv2.circle(canvas, p1, 6, (0, 0, 255), -1)
                cv2.circle(canvas, p2, 6, (255, 0, 0), -1)
                draw_arrow(canvas, ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2), ramp, (0, 165, 255), 30)

        gt, gc, gb = d.get("green", 0), d.get("green_center"), d.get("green_box")
        if gc and gb is not None and gt > 0:
            draw_box(canvas, gb, x1, y1, (0, 255, 0), 2)
            draw_arrow(canvas, (x1 + gc[0], y1 + gc[1]), gt, (0, 255, 0), 20)

        rt, rc, rb = d.get("redTube", 0), d.get("redTube_center"), d.get("redTube_box")
        if rc and rb is not None and rt > 0:
            draw_box(canvas, rb, x1, y1, (0, 0, 255), 2)
            draw_arrow(canvas, (x1 + rc[0], y1 + rc[1]), rt, (0, 0, 255), 20)

        bt, bc, bb = d.get("blueTube", 0), d.get("blueTube_center"), d.get("blueTube_box")
        if bc and bb is not None and bt > 0:
            draw_box(canvas, bb, x1, y1, (255, 0, 0), 2)
            draw_arrow(canvas, (x1 + bc[0], y1 + bc[1]), bt, (255, 0, 0), 20)

        if d.get("robot", 0) > 0:
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(canvas, (cx, cy), min(w, h) // 4, (0, 140, 255), 4)
            cv2.putText(canvas, "ROBOT", (x1 + 5, y1 + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

        cv2.putText(canvas, f"L:{lvl}", (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        info_y = y2 - 10
        if ramp > 0:
            cv2.putText(canvas, f"R:{ramp}({d.get('ramp_dir_precise', '')})", (x1 + 5, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            info_y -= 12
        if gt > 0:
            cv2.putText(canvas, f"G:{gt}", (x1 + 5, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            info_y -= 12
        if rt > 0: cv2.putText(canvas, f"RT:{rt}", (x1 + 35, info_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        if bt > 0: cv2.putText(canvas, f"BT:{bt}", (x1 + 35, info_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    hc, wc = canvas.shape[:2]
    if max(hc, wc) > max_dim:
        s = max_dim / max(hc, wc)
        canvas = cv2.resize(canvas, (int(wc * s), int(hc * s)), interpolation=cv2.INTER_AREA)
    cv2.imshow("Map Grid", canvas)

def visualize_map_schematic(clean_data: List[Dict], cell: int = 100, max_dim: int = 1000):
    margin = 40
    cw, ch = 8 * cell + margin * 2, 8 * cell + margin * 2
    canvas = np.ones((ch, cw, 3), dtype=np.uint8) * 240
    data_by_idx = {item['index']: item for item in clean_data}

    for r in range(8):
        for c in range(8):
            x1, y1 = margin + c * cell, margin + r * cell
            x2, y2 = x1 + cell, y1 + cell
            idx = r * 8 + c
            d = data_by_idx.get(idx, {})
            lvl = d.get("level", 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 60, 60) if lvl == 1 else (200, 200, 200), -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.putText(canvas, str(idx), (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255) if lvl == 1 else (0, 0, 0), 2)
            cx, cy = x1 + cell // 2, y1 + cell // 2
            cv2.line(canvas, (cx - 4, cy), (cx + 4, cy), (80, 80, 80), 1)
            cv2.line(canvas, (cx, cy - 4), (cx, cy + 4), (80, 80, 80), 1)

            ramp = d.get("ramp", 0)
            if ramp > 0:
                al = 35
                if ramp == 2: cv2.arrowedLine(canvas, (cx, cy - al), (cx, cy + al), (0, 165, 255), 4, tipLength=0.3)
                else: cv2.arrowedLine(canvas, (cx - al, cy), (cx + al, cy), (0, 165, 255), 4, tipLength=0.3)
                cv2.putText(canvas, f"R:{ramp}({d.get('ramp_dir_precise', '')})", (x1 + 5, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 2)

            gt = d.get("green", 0)
            if gt > 0:
                rw, rh = (40, 14) if gt == 1 else (14, 40)
                cv2.rectangle(canvas, (cx - rw//2, cy - rh//2), (cx + rw//2, cy + rh//2), (0, 255, 0), 3)
                cv2.putText(canvas, "G", (x2 - 20, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 2)

            rt = d.get("redTube", 0)
            if rt > 0:
                rw, rh = (44, 12) if rt == 1 else (12, 44)
                ox, oy = (-8 if rt == 2 else 0, -8 if rt == 1 else 0)
                cv2.rectangle(canvas, (cx - rw//2 + ox, cy - rh//2 + oy), (cx + rw//2 + ox, cy + rh//2 + oy), (0, 0, 255), 3)
                cv2.putText(canvas, "RT", (x1 + 5, y2 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

            bt = d.get("blueTube", 0)
            if bt > 0:
                rw, rh = (44, 12) if bt == 1 else (12, 44)
                ox, oy = (8 if bt == 2 else 0, 8 if bt == 1 else 0)
                cv2.rectangle(canvas, (cx - rw//2 + ox, cy - rh//2 + oy), (cx + rw//2 + ox, cy + rh//2 + oy), (255, 0, 0), 3)
                cv2.putText(canvas, "BT", (x2 - 25, y2 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)

            if d.get("robot", 0) > 0:
                pts = np.array([[cx, cy - 30], [cx - 20, cy + 15], [cx + 20, cy + 15]], dtype=np.int32)
                cv2.fillPoly(canvas, [pts], (0, 140, 255))
                cv2.polylines(canvas, [pts], True, (0, 0, 0), 2)
                cv2.putText(canvas, "RBT", (x1 + 5, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 200), 2)
            # if d.get("object_data"):
    cv2.putText(canvas, "Field Map Schematic", (margin, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(canvas, "R=Ramp G=Green RT=RedTube BT=BlueTube RBT=Robot", (margin, ch - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    hc, wc = canvas.shape[:2]
    if max(hc, wc) > max_dim:
        s = max_dim / max(hc, wc)
        canvas = cv2.resize(canvas, (int(wc * s), int(hc * s)), interpolation=cv2.INTER_AREA)
    cv2.imshow("Map Schematic", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def _json_serializer(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, tuple): return list(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

def run(sections: List[Tuple[int, int, int, np.ndarray]], out_dir: str, debug: bool = False, use_qr_flag: Optional[bool] = None) -> List[Dict]:
    map_data = []
    for idx, r, c, img in sections:
        a = analyze_section(img, r, c, use_qr_flag)
        map_data.append({
            "index": idx, "row": r, "col": c, "level": a["level"], "ramp": a["ramp"],
            "ramp_angle": a["ramp_angle"], "ramp_dir_precise": a["ramp_dir_precise"],
            "ramp_centers": a["ramp_centers"], "green": a["green"], "green_center": a["green_center"],
            "green_box": a["green_box"], "redTube": a["redTube"], "redTube_center": a["redTube_center"],
            "redTube_box": a["redTube_box"], "blueTube": a["blueTube"], "blueTube_center": a["blueTube_center"],
            "blueTube_box": a["blueTube_box"], "robot": a["robot"], "robot_area": a["robot_area"],
            "obj": a["obj"], "obj_pos": a["obj_pos"],
        })

    _, _, _, sample = sections[0]
    sec_h, sec_w = sample.shape[:2]
    fix_cut_objects(map_data, sec_w, sec_h)
    fix_robot_uniqueness(map_data)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "map_raw.json"), "w", encoding="utf-8") as f:
        json.dump(map_data, f, indent=2, ensure_ascii=False, default=_json_serializer)

    clean = []
    for item in map_data:
        clean.append({
            "index": item["index"], "row": item["row"], "col": item["col"],
            "level": item["level"], "ramp": item["ramp"], "ramp_angle": item["ramp_angle"],
            "ramp_dir_precise": item["ramp_dir_precise"], "green": item["green"],
            "redTube": item["redTube"], "blueTube": item["blueTube"], "robot": item["robot"],
            "obj": item["obj"], "obj_pos": item["obj_pos"],
        })
    with open(os.path.join(out_dir, "map.json"), "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=4, ensure_ascii=False)

    if debug:
        for item in clean:
            parts = [f"Lvl:{item['level']}"]
            if item['ramp'] > 0: parts.append(f"Ramp:{item['ramp']}({item['ramp_dir_precise']})")
            if item['green'] > 0: parts.append(f"Green:{item['green']}")
            if item['redTube'] > 0: parts.append(f"RT:{item['redTube']}")
            if item['blueTube'] > 0: parts.append(f"BT:{item['blueTube']}")
            if item['robot'] > 0: parts.append("ROBOT")
            if item['obj'] > 0: parts.append(f"OBJ[pos={item['obj_pos']}]")
            print(f"   [{item['index']:2d}]    " + " | ".join(parts))
        visualize_map_grid(map_data, sections)
        visualize_map_schematic(clean)

    return clean

def main():
    parser = argparse.ArgumentParser(description="Analysis of RRO field sections")
    parser.add_argument("--input-dir", default="output/sections")
    parser.add_argument("-o", "--out-dir", default="output")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--robot-mode", choices=["orange", "qr", "aruco"], default="qr", 
                        help="Robot detection mode: orange (default), qr (code 42), aruco (code 67)")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Directory not found: {args.input_dir}")
        sys.exit(1)

    sections = []
    for i in range(64):
        path = os.path.join(args.input_dir, f"section_{i:02d}.png")
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                sections.append((i, i // 8, i % 8, img))

    if not sections:
        print("No sections found.")
        sys.exit(1)

    use_qr_flag = None
    if args.robot_mode == "qr":
        use_qr_flag = True
    elif args.robot_mode == "aruco":
        use_qr_flag = False

    print(f"Loaded {len(sections)} sections. Robot mode: {args.robot_mode}")
    run(sections, args.out_dir, debug=args.debug, use_qr_flag=use_qr_flag)

if __name__ == "__main__":
    main()