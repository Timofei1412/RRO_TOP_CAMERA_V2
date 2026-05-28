"""
carry_planner.py — Планировщик перевозки труб для Future Engineers (RRO 2026).
Поддерживает многоэтажные поля с рампами.
Команды: F{n}, R, L, around, U, D, T, P (без ID)
Логика: F перед D и после U уменьшается на 1 (F0 удаляется).
"""
import itertools
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from router import (
    FieldRouter,
    HEADING_N, HEADING_E, HEADING_S, HEADING_W,
    HEADING_NAMES, HEADING_DELTA,
)

def _pt_rc(pt) -> Tuple[int, int]:
    return pt[0], pt[1]

def _count_turns(commands: List[str]) -> int:
    return sum(1 for c in commands if c in ("R", "L", "around"))

def _adjust_ramp_commands(commands: List[str]) -> List[str]:
    """Корректирует F: -1 перед D и -1 после U. F0 удаляется."""
    out = []
    for cmd in commands:
        if cmd == 'D':
            if out and out[-1].startswith('F'):
                n = int(out[-1][1:])
                if n > 1: out[-1] = f"F{n-1}"
                else: out.pop()
            out.append('D')
        elif cmd == 'U':
            out.append('U')
        elif cmd.startswith('F'):
            n = int(cmd[1:])
            if out and out[-1] == 'U':
                if n > 1: out.append(f"F{n-1}")
            else:
                out.append(cmd)
        else:
            out.append(cmd)
    return out

class CarryPlanner:
    def __init__(self, router: FieldRouter):
        self.router = router
        self.tubes: List[Dict] = []
        self.pods: List[Dict] = []
        self.prefer_straight: bool = False
        self._collect_objects()

    def _collect_objects(self):
        for (r, c), data in self.router.grid.items():
            level = data.get("level", 0)
            if data.get("redTube", 0) > 0:
                self.tubes.append({"id": f"RT_{r}_{c}", "color": "red", "row": r, "col": c, "type": data["redTube"], "level": level})
            if data.get("blueTube", 0) > 0:
                self.tubes.append({"id": f"BT_{r}_{c}", "color": "blue", "row": r, "col": c, "type": data["blueTube"], "level": level})
            if data.get("green", 0) > 0:
                self.pods.append({"id": f"G_{r}_{c}", "row": r, "col": c, "type": data["green"], "level": level})

    def _get_object_id(self, r: int, c: int) -> Optional[str]:
        data = self.router.grid.get((r, c), {})
        if data.get("redTube", 0) > 0: return f"RT_{r}_{c}"
        if data.get("blueTube", 0) > 0: return f"BT_{r}_{c}"
        if data.get("green", 0) > 0: return f"G_{r}_{c}"
        return None

    def _get_cell_level(self, r: int, c: int) -> int:
        data = self.router.grid.get((r, c))
        if not data: return 0
        return 1 if data.get("ramp", 0) > 0 else data.get("level", 0)

    def _approach_positions(self, obj: Dict) -> List[Tuple[int, int, int]]:
        r, c, t = obj["row"], obj["col"], obj["type"]
        if t == 2: return [(r - 1, c, HEADING_S), (r + 1, c, HEADING_N)]
        return [(r, c - 1, HEADING_E), (r, c + 1, HEADING_W)]

    def _is_cell_free(self, r: int, c: int, ignored_objects: Set[str]) -> bool:
        if not (0 <= r < 8 and 0 <= c < 8): return False
        data = self.router.grid.get((r, c))
        if not data: return False
        cell_level = self._get_cell_level(r, c)
        if (r, c, cell_level) not in self.router.nodes: return False
        obj_id = self._get_object_id(r, c)
        return obj_id is None or obj_id in ignored_objects

    def _validate_path(self, path, ignored_objects: Set[str]) -> bool:
        for pt in path:
            if not self._is_cell_free(pt[0], pt[1], ignored_objects): return False
        return True

    def _route(self, start_rc: Tuple[int, int], start_h: int, goal_rc: Tuple[int, int], goal_h: int, ignored_objects: Optional[Set[str]] = None):
        if ignored_objects is None: ignored_objects = set()
        if not self._is_cell_free(start_rc[0], start_rc[1], ignored_objects) or not self._is_cell_free(goal_rc[0], goal_rc[1], ignored_objects):
            return None
        try:
            result = self.router.find_path(start_rc, goal_rc, start_h, prefer_straight=self.prefer_straight)
        except Exception: return None
        if not result or len(result) != 3: return None
        commands, cost, path = result
        if not self._validate_path(path, ignored_objects): return None

        final_heading = start_h
        for cmd in commands:
            if cmd == "R": final_heading = (final_heading + 1) % 4
            elif cmd == "L": final_heading = (final_heading - 1) % 4
            elif cmd == "around": final_heading = (final_heading + 2) % 4

        diff = (goal_h - final_heading) % 4
        final_commands = list(commands)
        final_cost = float(cost)
        if diff == 1: final_commands.append("R"); final_cost += 0.5; final_heading = (final_heading + 1) % 4
        elif diff == 3: final_commands.append("L"); final_cost += 0.5; final_heading = (final_heading - 1) % 4
        elif diff == 2: final_commands.append("around"); final_cost += 1.0; final_heading = (final_heading + 2) % 4

        norm_path = [pt[:2] for pt in path]
        return final_commands, final_cost, norm_path, final_heading, _count_turns(final_commands)

    def _plan_special(self, start, start_heading, unique_tube, common_tubes, verbose=True) -> Optional[Dict]:
        pods = self.pods
        manhattan = lambda p1, p2: abs(p1['row'] - p2['row']) + abs(p1['col'] - p2['col'])
        middle_pod = min(pods, key=lambda p: sum(manhattan(p, o) for o in pods if o['id'] != p['id']))
        other_pods = [p for p in pods if p['id'] != middle_pod['id']]

        best, checked = None, 0
        for common_perm in itertools.permutations(common_tubes):
            pairs = [(unique_tube, middle_pod), (common_perm[0], other_pods[0]), (common_perm[1], other_pods[1])]
            for pairs_perm in itertools.permutations(pairs):
                tube_order, pod_perm = [p[0] for p in pairs_perm], [p[1] for p in pairs_perm]
                approach_choices = []
                valid = True
                for tube, pod in zip(tube_order, pod_perm):
                    t_appr = [a for a in self._approach_positions(tube) if self._is_cell_free(a[0], a[1], set())]
                    p_appr = [a for a in self._approach_positions(pod) if self._is_cell_free(a[0], a[1], set())]
                    if not t_appr or not p_appr: valid = False; break
                    approach_choices.append(list(itertools.product(t_appr, p_appr)))
                if not valid or not approach_choices: continue
                for combo in itertools.product(*approach_choices):
                    checked += 1
                    res = self._evaluate_plan(start, start_heading, list(tube_order), list(pod_perm), list(combo))
                    if res and (best is None or res['total_cost'] < best['total_cost']):
                        best = res
        return best

    def plan(self, start: Tuple[int, int], start_heading: int, verbose: bool = True, animate: bool = False,
             cell_size: int = 100, robot_start: Optional[Tuple[int, int]] = None, prefer_straight: bool = False) -> Optional[Dict]:
        self.prefer_straight = prefer_straight
        if robot_start is not None:
            start = robot_start
        if not self.tubes or not self.pods: return None
        
        # Максимально возможное число труб, которое можно увезти
        max_tubes = min(len(self.tubes), len(self.pods))
        if max_tubes == 0: return None
        
        if not self._is_cell_free(start[0], start[1], set()): return None

        if len(self.tubes) == 3 and len(self.pods) == 3:
            reds = [t for t in self.tubes if t['color'] == 'red']
            blues = [t for t in self.tubes if t['color'] == 'blue']
            unique, common = (reds[0], blues) if len(reds)==1 and len(blues)==2 else (blues[0], reds) if len(blues)==1 and len(reds)==2 else (None, [])
            if unique:
                special = self._plan_special(start, start_heading, unique, common, verbose)
                if special:
                    if animate: special["start_heading"] = start_heading; self.animate_plan(special, cell_size=cell_size)
                    return special

        best, checked = None, 0
        for num_tubes in range(max_tubes, 0, -1):
            if best is not None:
                # Нашли рабочий план с максимальным кол-вом труб, дальше искать нет смысла
                break
            for tube_sel in itertools.combinations(self.tubes, num_tubes):
                for pod_sel in itertools.combinations(self.pods, num_tubes):
                    for pod_perm in itertools.permutations(pod_sel):
                        for tube_ord in itertools.permutations(tube_sel):
                            appr = []
                            valid = True
                            for t, p in zip(tube_ord, pod_perm):
                                ta = [a for a in self._approach_positions(t) if self._is_cell_free(a[0], a[1], set())]
                                pa = [a for a in self._approach_positions(p) if self._is_cell_free(a[0], a[1], set())]
                                if not ta or not pa: valid = False; break
                                appr.append(list(itertools.product(ta, pa)))
                            if not valid or not appr: continue
                            for combo in itertools.product(*appr):
                                checked += 1
                                res = self._evaluate_plan(start, start_heading, list(tube_ord), list(pod_perm), list(combo))
                                if res and (best is None or res['total_cost'] < best['total_cost']):
                                    best = res

        if best and animate: best["start_heading"] = start_heading; self.animate_plan(best, cell_size=cell_size)
        return best

    def _evaluate_plan(self, start, start_heading, tube_order, pod_perm, combo) -> Optional[Dict]:
        all_commands, total_cost, total_turns, current_pos, current_heading = [], 0.0, 0, start, start_heading
        segments, ignored, full_raw_path = [], set(), [start]

        for (tube, pod, (t_appr, p_appr)) in zip(tube_order, pod_perm, combo):
            r1 = self._route(current_pos, current_heading, (t_appr[0], t_appr[1]), t_appr[2], ignored)
            if not r1: return None
            cmds1, cost1, path1, fh1, turns1 = r1
            if self._get_cell_level(t_appr[0], t_appr[1]) != tube['level']: return None
            all_commands.extend(cmds1); total_cost += cost1 + 0.5; total_turns += turns1
            full_raw_path.extend(path1[1:]); current_pos = (t_appr[0], t_appr[1]); current_heading = (fh1 + 2) % 4
            ignored.add(tube["id"]); all_commands.append("T")

            r2 = self._route(current_pos, current_heading, (p_appr[0], p_appr[1]), p_appr[2], ignored)
            if not r2: return None
            cmds2, cost2, path2, fh2, turns2 = r2
            if self._get_cell_level(p_appr[0], p_appr[1]) != pod['level']: return None
            all_commands.extend(cmds2); total_cost += cost2 + 0.5; total_turns += turns2
            full_raw_path.extend(path2[1:]); current_pos = (p_appr[0], p_appr[1]); current_heading = (fh2 + 2) % 4
            ignored.add(pod["id"]); all_commands.append("P")
            segments.append({"tube": tube, "pod": pod, "tube_approach": t_appr, "pod_approach": p_appr, "to_tube_cost": cost1, "to_pod_cost": cost2})

        all_commands = _adjust_ramp_commands(all_commands)

        return {"commands": all_commands, "total_cost": total_cost, "total_turns": total_turns, "segments": segments,
                "raw_path": full_raw_path, "final_pos": current_pos, "final_heading": current_heading,
                "final_level": self._get_cell_level(current_pos[0], current_pos[1])}

    def print_plan(self, plan: Dict):
        for i, cmd in enumerate(plan["commands"], 1): print(f"   {i:2d}. {cmd}")

    def animate_plan(self, plan: Dict, cell_size: int = 100, step_delay: float = 0.04, pause_event: float = 0.6):
        rows, cols, margin = 8, 8, 60
        h, w = rows * cell_size + margin * 2 + 60, cols * cell_size + margin * 2
        window_name = "Carry Animation"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        picked_up, placed, current_cargo = set(), {}, None
        events_at_idx, cursor = {}, 0
        raw = plan["raw_path"]
        for seg in plan["segments"]:
            ta, pa = (seg["tube_approach"][0], seg["tube_approach"][1]), (seg["pod_approach"][0], seg["pod_approach"][1])
            for j in range(cursor, len(raw)):
                if _pt_rc(raw[j]) == ta: events_at_idx[j] = ("T", seg["tube"]); cursor = j + 1; break
            for j in range(cursor, len(raw)):
                if _pt_rc(raw[j]) == pa: events_at_idx[j] = ("P", seg["pod"], seg["tube"]); cursor = j + 1; break

        L, H, W_tri = 34, 16, 22
        tri_tpl = {HEADING_N: [(0,-L),(-W_tri,H),(W_tri,H)], HEADING_E: [(L,0),(-H,-W_tri),(-H,W_tri)],
                   HEADING_S: [(0,L),(W_tri,-H),(-W_tri,-H)], HEADING_W: [(-L,0),(H,W_tri),(H,-W_tri)]}
        cc = lambda r,c: (margin + c*cell_size + cell_size//2, margin + r*cell_size + cell_size//2)

        def draw_bg(c):
            c[:] = 240
            for r in range(rows):
                for col in range(cols):
                    x1,y1 = margin+col*cell_size, margin+r*cell_size
                    d = self.router.grid.get((r,col),{})
                    bg = (60,60,60) if (d.get("level",0)==1 or d.get("ramp",0) >0) else (210,210,210)
                    cv2.rectangle(c,(x1,y1),(x1+cell_size,y1+cell_size),bg,-1); cv2.rectangle(c,(x1,y1),(x1+cell_size,y1+cell_size),(0,0,0),1)
                    cv2.putText(c,str(r*8+col),(x1+4,y1+18),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0) if bg==(210,210,210) else (255,255,255),1)
                    if d.get("ramp",0) >0:
                        cx,cy = x1+cell_size//2, y1+cell_size//2
                        col_ln = (0,165,255)
                        if d["ramp"]==2: cv2.arrowedLine(c,(cx,cy-25),(cx,cy+25),col_ln,3,line_type=cv2.LINE_AA,tipLength=0.3)
                        else: cv2.arrowedLine(c,(cx-25,cy),(cx+25,cy),col_ln,3,line_type=cv2.LINE_AA,tipLength=0.3)

        def draw_obj(c):
            for t in self.tubes:
                if t["id"] in picked_up: continue
                cx,cy = cc(t["row"],t["col"]); clr=(0,0,230) if t["color"]=="red" else (230,0,0)
                if t["type"]==1: cv2.rectangle(c,(cx-6,cy-22),(cx+6,cy+22),clr,3)
                else: cv2.rectangle(c,(cx-22,cy-6),(cx+22,cy+6),clr,3)
            for p in self.pods:
                cx,cy = cc(p["row"],p["col"]); cv2.circle(c,(cx+22,cy+22),10,(0,200,0),3)
                if p["id"] in placed:
                    clr=(0,0,220) if placed[p["id"]]=="red" else (220,0,0)
                    if p["type"]==1: cv2.rectangle(c,(cx-5,cy-18),(cx+5,cy+18),clr,-1)
                    else: cv2.rectangle(c,(cx-18,cy-5),(cx+18,cy+5),clr,-1)

        def draw_rbt(c, px, py, h):
            pts = np.array([[int(px+dx), int(py+dy)] for dx,dy in tri_tpl.get(h, tri_tpl[HEADING_N])], dtype=np.int32)
            cv2.fillConvexPoly(c, pts, (0,230,230)); cv2.polylines(c, [pts], True, (0,0,0), 2)
            if current_cargo: cv2.circle(c,(int(px),int(py)),7,(0,0,220) if current_cargo["color"]=="red" else (220,0,0),-1)

        canvas = np.ones((h, w, 3), dtype=np.uint8) * 240
        heading = plan.get("start_heading", HEADING_N)
        cmd_idx = 0

        for i in range(len(raw) - 1):
            p1, p2 = _pt_rc(raw[i]), _pt_rc(raw[i+1])
            if p2[0] < p1[0]: heading = HEADING_N
            elif p2[0] > p1[0]: heading = HEADING_S
            elif p2[1] > p1[1]: heading = HEADING_E
            else: heading = HEADING_W

            for s in range(max(6, int(cell_size/8))):
                t = (s+1)/max(6, int(cell_size/8))
                px = cc(*p1)[0] + (cc(*p2)[0] - cc(*p1)[0]) * t
                py = cc(*p1)[1] + (cc(*p2)[1] - cc(*p1)[1]) * t
                draw_bg(canvas); draw_obj(canvas); draw_rbt(canvas, px, py, heading)
                cur_cmd = plan["commands"][cmd_idx] if cmd_idx < len(plan["commands"]) else "-"
                cv2.putText(canvas, f"Cmd: {cur_cmd} | InHand: {current_cargo['id'] if current_cargo else '-'}", (margin, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,230,255), 1)
                cv2.imshow(window_name, canvas)
                if cv2.waitKey(int(step_delay*1000)) & 0xFF in (27, ord('q')): cv2.destroyWindow(window_name); return

            if (i+1) in events_at_idx:
                ev = events_at_idx[i+1]
                if ev[0] == "T": picked_up.add(ev[1]["id"]); current_cargo = ev[1]
                elif ev[0] == "P": placed[ev[1]["id"]] = ev[2]["color"]; current_cargo = None
                while cmd_idx < len(plan["commands"]) and plan["commands"][cmd_idx] in ("T", "P"): cmd_idx += 1
                for _ in range(6):
                    draw_bg(canvas); draw_obj(canvas); draw_rbt(canvas, *cc(*p2), heading)
                    cv2.putText(canvas, f"EVENT: {ev[0]}", (margin, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                    cv2.imshow(window_name, canvas)
                    if cv2.waitKey(int(pause_event*1000/6)) & 0xFF in (27, ord('q')): cv2.destroyWindow(window_name); return
            else:
                while cmd_idx < len(plan["commands"]) and plan["commands"][cmd_idx] not in ("T", "P"):
                    cmd_idx += 1

        draw_bg(canvas); draw_obj(canvas); draw_rbt(canvas, *cc(*_pt_rc(raw[-1])), heading)
        cv2.putText(canvas, f"DONE! Placed: {len(placed)}/{len(self.pods)} | Cost: {plan['total_cost']:.2f}", (margin, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,230,255), 1)
        cv2.imshow(window_name, canvas)
        cv2.waitKey(0); cv2.destroyWindow(window_name)

def commands_to_short_carry(commands: List[str]) -> str:
    return " -> ".join(commands) if commands else ""