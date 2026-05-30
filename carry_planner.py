import itertools
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from router import (
    FieldRouter,
    HEADING_N, HEADING_E, HEADING_S, HEADING_W,
    HEADING_NAMES, HEADING_DELTA,
)


def _to_rc(pt: Tuple[int, int]) -> Tuple[int, int]:
    return pt[0], pt[1]


def _count_turns(cmds: List[str]) -> int:
    return sum(1 for c in cmds if c in ("R", "L", "around"))


def _fix_ramp_cmds(cmds: List[str]) -> List[str]:
    """Уменьшает F перед D и удаляет F0 после U."""
    out = []
    for cmd in cmds:
        if cmd == "D":
            if out and out[-1].startswith("F"):
                n = int(out[-1][1:])
                if n > 1:
                    out[-1] = f"F{n-1}"
                else:
                    out.pop()
            out.append("D")
        elif cmd == "U":
            out.append("U")
        elif cmd.startswith("F"):
            n = int(cmd[1:])
            if out and out[-1] == "U":
                if n > 1:
                    out.append(f"F{n-1}")
                else:
                    out.append(cmd)
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
        self.prefer_straight = False
        self._collect_objects()

    def _collect_objects(self):
        for (r, c), data in self.router.grid.items():
            level = data.get("level", 0)
            if data.get("redTube", 0) > 0:
                self.tubes.append({
                    "id": f"RT_{r}_{c}", "color": "red",
                    "row": r, "col": c, "type": data["redTube"], "level": level
                })
            if data.get("blueTube", 0) > 0:
                self.tubes.append({
                    "id": f"BT_{r}_{c}", "color": "blue",
                    "row": r, "col": c, "type": data["blueTube"], "level": level
                })
            if data.get("green", 0) > 0:
                self.pods.append({
                    "id": f"G_{r}_{c}", "row": r, "col": c,
                    "type": data["green"], "level": level
                })

    def _get_obj_id(self, r: int, c: int) -> Optional[str]:
        data = self.router.grid.get((r, c), {})
        if data.get("redTube", 0) > 0: return f"RT_{r}_{c}"
        if data.get("blueTube", 0) > 0: return f"BT_{r}_{c}"
        if data.get("green", 0) > 0: return f"G_{r}_{c}"
        return None

    def _cell_level(self, r: int, c: int) -> int:
        data = self.router.grid.get((r, c))
        if not data: return 0
        return 1 if data.get("ramp", 0) > 0 else data.get("level", 0)

    def _approach_cells(self, obj: Dict) -> List[Tuple[int, int, int]]:
        r, c, t = obj["row"], obj["col"], obj["type"]
        if t == 2:
            return [(r - 1, c, HEADING_S), (r + 1, c, HEADING_N)]
        return [(r, c - 1, HEADING_E), (r, c + 1, HEADING_W)]

    def _is_free(self, r: int, c: int, ignore: Set[str]) -> bool:
        if not (0 <= r < 8 and 0 <= c < 8): return False
        data = self.router.grid.get((r, c))
        if not data: return False
        lvl = self._cell_level(r, c)
        if (r, c, lvl) not in self.router.nodes: return False
        oid = self._get_obj_id(r, c)
        return oid is None or oid in ignore

    def _validate_path(self, path: List[Tuple], ignore: Set[str]) -> bool:
        for pt in path:
            if not self._is_free(pt[0], pt[1], ignore):
                return False
        return True

    def _route(self, start_rc, start_h, goal_rc, goal_h, ignore: Optional[Set[str]] = None):
        if ignore is None: ignore = set()
        if not self._is_free(start_rc[0], start_rc[1], ignore) or \
           not self._is_free(goal_rc[0], goal_rc[1], ignore):
            return None

        try:
            res = self.router.find_path(start_rc, goal_rc, start_h, prefer_straight=self.prefer_straight)
        except Exception:
            return None

        if not res or len(res) != 3: return None
        cmds, cost, path = res
        if not self._validate_path(path, ignore): return None

        cur_h = start_h
        for cmd in cmds:
            if cmd == "R": cur_h = (cur_h + 1) % 4
            elif cmd == "L": cur_h = (cur_h - 1) % 4
            elif cmd == "around": cur_h = (cur_h + 2) % 4

        diff = (goal_h - cur_h) % 4
        final_cmds = list(cmds)
        final_cost = float(cost)

        if diff == 1:
            final_cmds.append("R"); final_cost += 0.5; cur_h = (cur_h + 1) % 4
        elif diff == 3:
            final_cmds.append("L"); final_cost += 0.5; cur_h = (cur_h - 1) % 4
        elif diff == 2:
            final_cmds.append("around"); final_cost += 1.0; cur_h = (cur_h + 2) % 4

        norm_path = [pt[:2] for pt in path]
        return final_cmds, final_cost, norm_path, cur_h, _count_turns(final_cmds)

    def _plan_special(self, start, start_h, unique_tube, common_tubes, verbose=True):
        pods = self.pods
        manhattan = lambda p1, p2: abs(p1["row"] - p2["row"]) + abs(p1["col"] - p2["col"])
        mid_pod = min(pods, key=lambda p: sum(manhattan(p, o) for o in pods if o["id"] != p["id"]))
        other_pods = [p for p in pods if p["id"] != mid_pod["id"]]

        best, checked = None, 0
        for perm in itertools.permutations(common_tubes):
            pairs = [(unique_tube, mid_pod), (perm[0], other_pods[0]), (perm[1], other_pods[1])]
            for pair_perm in itertools.permutations(pairs):
                t_order = [p[0] for p in pair_perm]
                p_perm = [p[1] for p in pair_perm]

                choices = []
                valid = True
                for t, p in zip(t_order, p_perm):
                    ta = [a for a in self._approach_cells(t) if self._is_free(a[0], a[1], set())]
                    pa = [a for a in self._approach_cells(p) if self._is_free(a[0], a[1], set())]
                    if not ta or not pa:
                        valid = False
                        break
                    choices.append(list(itertools.product(ta, pa)))

                if not valid or not choices: continue

                for combo in itertools.product(*choices):
                    checked += 1
                    res = self._eval_plan(start, start_h, list(t_order), list(p_perm), list(combo))
                    if res and (best is None or res["total_cost"] < best["total_cost"]):
                        best = res
        return best

    def plan(self, start, start_h, verbose=True, animate=False, cell_size=100,
             robot_start=None, prefer_straight=False):
        self.prefer_straight = prefer_straight
        if robot_start is not None:
            start = robot_start
        if not self.tubes or not self.pods: return None

        max_t = min(len(self.tubes), len(self.pods))
        if max_t == 0: return None
        if not self._is_free(start[0], start[1], set()): return None

        # Оптимизация для 3 труб/подставок
        if len(self.tubes) == 3 and len(self.pods) == 3:
            reds = [t for t in self.tubes if t["color"] == "red"]
            blues = [t for t in self.tubes if t["color"] == "blue"]
            unique, common = (reds[0], blues) if len(reds) == 1 else \
                             (blues[0], reds) if len(blues) == 1 else (None, [])
            if unique:
                special = self._plan_special(start, start_h, unique, common, verbose)
                if special:
                    if animate:
                        special["start_heading"] = start_h
                        self.animate_plan(special, cell_size=cell_size)
                    return special

        best, checked = None, 0
        for num in range(max_t, 0, -1):
            if best is not None: break
            for t_sel in itertools.combinations(self.tubes, num):
                for p_sel in itertools.combinations(self.pods, num):
                    for p_perm in itertools.permutations(p_sel):
                        for t_ord in itertools.permutations(t_sel):
                            appr = []
                            valid = True
                            for t, p in zip(t_ord, p_perm):
                                ta = [a for a in self._approach_cells(t) if self._is_free(a[0], a[1], set())]
                                pa = [a for a in self._approach_cells(p) if self._is_free(a[0], a[1], set())]
                                if not ta or not pa:
                                    valid = False
                                    break
                                appr.append(list(itertools.product(ta, pa)))

                            if not valid or not appr: continue
                            for combo in itertools.product(*appr):
                                checked += 1
                                res = self._eval_plan(start, start_h, list(t_ord), list(p_perm), list(combo))
                                if res and (best is None or res["total_cost"] < best["total_cost"]):
                                    best = res

        if best and animate:
            best["start_heading"] = start_h
            self.animate_plan(best, cell_size=cell_size)
        return best

    def _eval_plan(self, start, start_h, tube_ord, pod_perm, combo):
        cmds, cost, turns = [], 0.0, 0
        cur_pos, cur_h = start, start_h
        ignored = set()
        path = [start]
        segs = []

        for tube, pod, (ta, pa) in zip(tube_ord, pod_perm, combo):
            r1 = self._route(cur_pos, cur_h, (ta[0], ta[1]), ta[2], ignored)
            if not r1: return None
            c1, cost1, p1, fh1, t1 = r1
            if self._cell_level(ta[0], ta[1]) != tube["level"]: return None

            cmds.extend(c1); cost += cost1 + 0.5; turns += t1
            path.extend(p1[1:])
            cur_pos = (ta[0], ta[1]); cur_h = (fh1 + 2) % 4
            ignored.add(tube["id"]); cmds.append("T")

            r2 = self._route(cur_pos, cur_h, (pa[0], pa[1]), pa[2], ignored)
            if not r2: return None
            c2, cost2, p2, fh2, t2 = r2
            if self._cell_level(pa[0], pa[1]) != pod["level"]: return None

            cmds.extend(c2); cost += cost2 + 0.5; turns += t2
            path.extend(p2[1:])
            cur_pos = (pa[0], pa[1]); cur_h = (fh2 + 2) % 4
            ignored.add(pod["id"]); cmds.append("P")
            segs.append({
                "tube": tube, "pod": pod,
                "tube_approach": ta, "pod_approach": pa,
                "to_tube_cost": cost1, "to_pod_cost": cost2
            })

        cmds = _fix_ramp_cmds(cmds)
        return {
            "commands": cmds, "total_cost": cost, "total_turns": turns,
            "segments": segs, "raw_path": path,
            "final_pos": cur_pos, "final_heading": cur_h,
            "final_level": self._cell_level(cur_pos[0], cur_pos[1])
        }

    def print_plan(self, plan: Dict):
        for i, cmd in enumerate(plan["commands"], 1):
            print(f"   {i:2d}. {cmd}")

    def animate_plan(self, plan: Dict, cell_size=100, delay=0.04, pause=0.6):
        rows, cols, m = 8, 8, 60
        h = rows * cell_size + m * 2 + 60
        w = cols * cell_size + m * 2
        win = "Carry Animation"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        picked, placed, cargo = set(), {}, None
        events, cur = {}, 0
        raw = plan["raw_path"]
        for seg in plan["segments"]:
            ta = (seg["tube_approach"][0], seg["tube_approach"][1])
            pa = (seg["pod_approach"][0], seg["pod_approach"][1])
            for j in range(cur, len(raw)):
                if _to_rc(raw[j]) == ta: events[j] = ("T", seg["tube"]); cur = j + 1; break
            for j in range(cur, len(raw)):
                if _to_rc(raw[j]) == pa: events[j] = ("P", seg["pod"], seg["tube"]); cur = j + 1; break

        L, H_tri, W_tri = 34, 16, 22
        tri_tpl = {
            HEADING_N: [(0,-L),(-W_tri,H_tri),(W_tri,H_tri)],
            HEADING_E: [(L,0),(-H_tri,-W_tri),(-H_tri,W_tri)],
            HEADING_S: [(0,L),(W_tri,-H_tri),(-W_tri,-H_tri)],
            HEADING_W: [(-L,0),(H_tri,W_tri),(H_tri,-W_tri)]
        }
        cc = lambda r, c: (m + c*cell_size + cell_size//2, m + r*cell_size + cell_size//2)

        def draw_bg(c):
            c[:] = 240
            for r in range(rows):
                for cl in range(cols):
                    x1, y1 = m + cl*cell_size, m + r*cell_size
                    d = self.router.grid.get((r, cl), {})
                    bg = (60, 60, 60) if (d.get("level", 0) == 1 or d.get("ramp", 0) > 0) else (210, 210, 210)
                    cv2.rectangle(c, (x1, y1), (x1+cell_size, y1+cell_size), bg, -1)
                    cv2.rectangle(c, (x1, y1), (x1+cell_size, y1+cell_size), (0, 0, 0), 1)
                    tc = (0, 0, 0) if bg == (210, 210, 210) else (255, 255, 255)
                    cv2.putText(c, str(r*8+cl), (x1+4, y1+18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, tc, 1)
                    if d.get("ramp", 0) > 0:
                        cx, cy = x1+cell_size//2, y1+cell_size//2
                        rd = d.get("ramp_dir_precise", 0)
                        col = (0, 165, 255)
                        if rd == "S": cv2.arrowedLine(c, (cx, cy-25), (cx, cy+25), col, 3, line_type=cv2.LINE_AA, tipLength=0.3)
                        elif rd == "N": cv2.arrowedLine(c, (cx, cy+25), (cx, cy-25), col, 3, line_type=cv2.LINE_AA, tipLength=0.3)
                        elif rd == "E": cv2.arrowedLine(c, (cx-25, cy), (cx+25, cy), col, 3, line_type=cv2.LINE_AA, tipLength=0.3)
                        elif rd == "W": cv2.arrowedLine(c, (cx+25, cy), (cx-25, cy), col, 3, line_type=cv2.LINE_AA, tipLength=0.3)

        def draw_objs(c):
            for t in self.tubes:
                if t["id"] in picked: continue
                cx, cy = cc(t["row"], t["col"])
                clr = (0, 0, 230) if t["color"] == "red" else (230, 0, 0)
                if t["type"] == 1: cv2.rectangle(c, (cx-6, cy-22), (cx+6, cy+22), clr, 3)
                else: cv2.rectangle(c, (cx-22, cy-6), (cx+22, cy+6), clr, 3)
            for p in self.pods:
                cx, cy = cc(p["row"], p["col"])
                cv2.circle(c, (cx+22, cy+22), 10, (0, 200, 0), 3)
                if p["id"] in placed:
                    clr = (0, 0, 220) if placed[p["id"]] == "red" else (220, 0, 0)
                    if p["type"] == 1: cv2.rectangle(c, (cx-5, cy-18), (cx+5, cy+18), clr, -1)
                    else: cv2.rectangle(c, (cx-18, cy-5), (cx+18, cy+5), clr, -1)

        def draw_bot(c, px, py, hd):
            pts = np.array([[int(px+dx), int(py+dy)] for dx, dy in tri_tpl.get(hd, tri_tpl[HEADING_N])], dtype=np.int32)
            cv2.fillConvexPoly(c, pts, (0, 230, 230))
            cv2.polylines(c, [pts], True, (0, 0, 0), 2)
            if cargo:
                cc = (0, 0, 220) if cargo["color"] == "red" else (220, 0, 0)
                cv2.circle(c, (int(px), int(py)), 7, cc, -1)

        canvas = np.ones((h, w, 3), dtype=np.uint8) * 240
        heading = plan.get("start_heading", HEADING_N)
        c_idx = 0

        for i in range(len(raw) - 1):
            p1, p2 = _to_rc(raw[i]), _to_rc(raw[i+1])
            if p2[0] < p1[0]: heading = HEADING_N
            elif p2[0] > p1[0]: heading = HEADING_S
            elif p2[1] > p1[1]: heading = HEADING_E
            else: heading = HEADING_W

            steps = max(6, int(cell_size / 8))
            for s in range(steps):
                t = (s + 1) / steps
                px = cc(*p1)[0] + (cc(*p2)[0] - cc(*p1)[0]) * t
                py = cc(*p1)[1] + (cc(*p2)[1] - cc(*p1)[1]) * t
                draw_bg(canvas); draw_objs(canvas); draw_bot(canvas, px, py, heading)
                cur_cmd = plan["commands"][c_idx] if c_idx < len(plan["commands"]) else "-"
                cargo_txt = cargo["id"] if cargo else "-"
                cv2.putText(canvas, f"Cmd: {cur_cmd} | InHand: {cargo_txt}", (m, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 1)
                cv2.imshow(win, canvas)
                if cv2.waitKey(int(delay * 1000)) & 0xFF in (27, ord("q")):
                    cv2.destroyWindow(win); return

            if (i + 1) in events:
                ev = events[i + 1]
                if ev[0] == "T": picked.add(ev[1]["id"]); cargo = ev[1]
                elif ev[0] == "P": placed[ev[1]["id"]] = ev[2]["color"]; cargo = None
                while c_idx < len(plan["commands"]) and plan["commands"][c_idx] in ("T", "P"): c_idx += 1
                for _ in range(6):
                    draw_bg(canvas); draw_objs(canvas); draw_bot(canvas, *cc(*p2), heading)
                    cv2.putText(canvas, f"EVENT: {ev[0]}", (m, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.imshow(win, canvas)
                    if cv2.waitKey(int(pause * 1000 / 6)) & 0xFF in (27, ord("q")):
                        cv2.destroyWindow(win); return
            else:
                while c_idx < len(plan["commands"]) and plan["commands"][c_idx] not in ("T", "P"):
                    c_idx += 1

        draw_bg(canvas); draw_objs(canvas); draw_bot(canvas, *cc(*_to_rc(raw[-1])), heading)
        cv2.putText(canvas, f"DONE! Placed: {len(placed)}/{len(self.pods)} | Cost: {plan['total_cost']:.2f}", (m, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 1)
        cv2.imshow(win, canvas)
        cv2.waitKey(0); cv2.destroyWindow(win)


def commands_to_short_carry(cmds: List[str]) -> str:
    return " -> ".join(cmds) if cmds else ""