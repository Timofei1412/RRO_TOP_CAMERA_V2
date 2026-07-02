import heapq
import json
import os
import argparse
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

HEADING_N, HEADING_E, HEADING_S, HEADING_W = 0, 1, 2, 3
HEADING_NAMES = {0: "N", 1: "E", 2: "S", 3: "W"}
HEADING_DELTA = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
RAMP_COST = 20
MOVE_COST = 1.0

NO_USE_RAMP = False

class FieldRouter:
    def __init__(self, map_data: List[Dict], debug: bool = False, noRamps: bool = False):
        self.map_data = map_data
        self.grid = {(item["row"], item["col"]): item for item in map_data}
        self.nodes, self.edges, self.blocked, self.ramp_info = set(), {}, {}, {}
        self.debug = debug

        self._build_graph()

    def _is_blocked(self, r: int, c: int, level: int) -> Tuple[bool, str]:
        data = self.grid.get((r, c))
        if not data: return True, "outside"
        if data.get("redTube", 0) > 0 or data.get("blueTube", 0) > 0 or data.get("green", 0) > 0:
            return True, "obstacle"
        if data.get("ramp", 0) > 0:
            return False, ""
        return (level != data.get("level", 0)), "level_mismatch"

    def _is_valid_ramp(self, r: int, c: int, ramp_dir: str) -> bool:
        
        if NO_USE_RAMP:
            return False
        
        if ramp_dir not in ("N", "S", "E", "W"): return False
        hd = {"N": HEADING_N, "E": HEADING_E, "S": HEADING_S, "W": HEADING_W}
        down_h = hd[ramp_dir]
        up_h = (down_h + 2) % 4

        currentData = self.grid.get((r, c))

        dr_d, dc_d = HEADING_DELTA[down_h]
        r_d, c_d = r + dr_d, c + dc_d
        down_data = self.grid.get((r_d, c_d))

        dr_u, dc_u = HEADING_DELTA[up_h]
        r_u, c_u = r + dr_u, c + dc_u
        up_data = self.grid.get((r_u, c_u))

        if not down_data or not up_data: return False # если за границей
        if down_data.get("level", 0) != 0 or down_data.get("ramp", 0) > 0: return False # спускаемся вверх или там пандус
        
        
        if up_data.get("level", 0) != 1 and up_data.get("ramp", 0) == 0: return False # сверху второй этаж без пандуса

        if up_data.get("ramp", 0) > 0:
            
            up_ramp_dir = up_data.get("ramp_dir_precise", "")
            if up_ramp_dir in hd:
                up_down_h = hd[up_ramp_dir]
                # Направление вниз соседнего пандуса должно смотреть на нас
                h = (up_down_h + 2) % 4
                up_dr, up_dc = HEADING_DELTA[h]
                up_r_d, up_c_d = r_u + up_dr, c_u + up_dc
                if (up_r_d, up_c_d) != (r, c):
                    return False
            else:
                return False
        return True

    def _build_graph(self):
        potential = set()
        for (r, c), data in self.grid.items():
            if data.get("ramp", 0) > 0:
                rd = data.get("ramp_dir_precise")
                if not self._is_valid_ramp(r, c, rd):
                    self.blocked[(r, c, 1)] = "invalid_ramp"
                    self.blocked[(r, c, 0)] = "invalid_ramp"
                    continue
                hd = {"N": HEADING_N, "E": HEADING_E, "S": HEADING_S, "W": HEADING_W}
                down_h = hd[rd]
                up_h = (down_h + 2) % 4
                self.ramp_info[(r, c)] = (down_h, up_h)
                potential.add((r, c, 1))
                self.blocked[(r, c, 0)] = "under_ramp"
            else:
                level = data.get("level", 0)
                node = (r, c, level)
                blocked, reason = self._is_blocked(r, c, level)
                if not blocked:
                    potential.add(node)
                else:
                    self.blocked[node] = reason

        self.nodes = set(potential)
        for node in self.nodes:
            self.edges[node] = []
            r, c, level = node
            data = self.grid.get((r, c))
            has_ramp = data.get("ramp", 0) > 0

            if has_ramp and level == 1:
                down_h, up_h = self.ramp_info.get((r, c), (None, None))
                if down_h is None: continue
                dr_d, dc_d = HEADING_DELTA[down_h]
                nb_down = (r + dr_d, c + dc_d, 0)
                if nb_down in self.nodes:
                    self.edges[node].append((nb_down, RAMP_COST, ("ramp_down", down_h)))
                dr_u, dc_u = HEADING_DELTA[up_h]
                nb_up = (r + dr_u, c + dc_u, 1)
                if nb_up in self.nodes:
                    self.edges[node].append((nb_up, MOVE_COST, ("move", up_h)))
            elif level == 0:
                for h, (dr, dc) in HEADING_DELTA.items():
                    nr, nc = r + dr, c + dc
                    nb = (nr, nc, 0)
                    if nb in self.nodes:
                        self.edges[node].append((nb, MOVE_COST, ("move", h)))
                    nd = self.grid.get((nr, nc))
                    if nd and nd.get("ramp", 0) > 0:
                        info = self.ramp_info.get((nr, nc))
                        if info:
                            _, up_h = info
                            if h == up_h:
                                nb_ramp = (nr, nc, 1)
                                if nb_ramp in self.nodes:
                                    self.edges[node].append((nb_ramp, RAMP_COST, ("ramp_up", h)))
            elif level == 1:
                for h, (dr, dc) in HEADING_DELTA.items():
                    nr, nc = r + dr, c + dc
                    nd = self.grid.get((nr, nc))
                    if nd and nd.get("ramp", 0) > 0:
                        info = self.ramp_info.get((nr, nc))
                        if info:
                            down_h, _ = info
                            if h == down_h:
                                nb_ramp = (nr, nc, 1)
                                if nb_ramp in self.nodes:
                                    self.edges[node].append((nb_ramp, MOVE_COST, ("move", h)))
                            continue
                    nb = (nr, nc, 1)
                    if nb in self.nodes:
                        self.edges[node].append((nb, MOVE_COST, ("move", h)))

    def _heuristic(self, a: Tuple, b: Tuple) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]) * 2

    def find_path(self, start, goal, start_h=HEADING_N, goal_h=None, prefer_straight=False):
        sd, gd = self.grid.get(start), self.grid.get(goal)
        if not sd or not gd: return None
        sl = 1 if sd.get("ramp", 0) > 0 else sd.get("level", 0)
        gl = 1 if gd.get("ramp", 0) > 0 else gd.get("level", 0)
        sn = (start[0], start[1], sl)
        gn = (goal[0], goal[1], gl)
        if sn not in self.nodes or gn not in self.nodes: return None

        counter = 0
        pq = []
        state = (*sn, start_h)

        if prefer_straight:
            heapq.heappush(pq, (0.0, 0, 0.0, counter, state))
            g = {state: (0.0, 0)}
        else:
            heapq.heappush(pq, (0.0, 0.0, counter, state))
            g = {state: 0.0}

        came = {}

        while pq:
            if prefer_straight:
                _, t, val, _, cur = heapq.heappop(pq)
                if (val, t) > g.get(cur, (float("inf"), float("inf"))): continue
            else:
                _, val, _, cur = heapq.heappop(pq)
                if val > g.get(cur, float("inf")): continue

            cn, ch = cur[:3], cur[3]
            if cn == gn:
                if goal_h is None or ch == goal_h:
                    tc = g[cur][0] if prefer_straight else g[cur]
                    return self._reconstruct_path(came, cur, state, tc)

            for nb, cost, act in self.edges.get(cn, []):
                atype, ah = act
                nxt = (*nb, ah)
                tg = (val if isinstance(val, float) else val[0]) + cost

                if prefer_straight:
                    tt = t + (1 if ah != ch else 0)
                    if (tg, tt) < g.get(nxt, (float("inf"), float("inf"))):
                        came[nxt] = (cur, act, cost)
                        g[nxt] = (tg, tt)
                        h = self._heuristic(nb, gn)
                        counter += 1
                        heapq.heappush(pq, (tg + h, tt, tg, counter, nxt))
                else:
                    if tg < g.get(nxt, float("inf")):
                        came[nxt] = (cur, act, cost)
                        g[nxt] = tg
                        h = self._heuristic(nb, gn)
                        counter += 1
                        heapq.heappush(pq, (tg + h, tg, counter, nxt))
        return None

    def _reconstruct_path(self, came, end, start, tc):
        st, acts = [end], []
        cur = end
        while cur != start:
            prev, act, _ = came[cur]
            acts.append(act)
            st.append(prev)
            cur = prev
        st.reverse()
        acts.reverse()
        raw = [s[:3] for s in st]
        cmds = self._acts_to_cmds(st, acts)
        return cmds, tc, raw

    def _acts_to_cmds(self, states, actions) -> List[str]:
        if not actions: return []
        cmds = []
        h = states[0][3]
        i = 0
        while i < len(actions):
            atype, ah = actions[i]
            if ah != h:
                d = (ah - h) % 4
                if d == 1: cmds.append("R")
                elif d == 3: cmds.append("L")
                elif d == 2: cmds.append("around")
                h = ah
            if atype == "move":
                n = 1; i += 1
                while i < len(actions) and actions[i] == ("move", h):
                    n += 1; i += 1
                cmds.append(f"F{n}")
            elif atype == "ramp_up":
                cmds.append("U"); h = ah; i += 1
            elif atype == "ramp_down":
                cmds.append("D"); h = ah; i += 1
        return cmds

    def save_graph(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)

        def edge_key(n):
            return f"{n[0]},{n[1]},{n[2]}"

        def ramp_key(n):
            return f"{n[0]},{n[1]}"

        edges = {}
        for n, eds in self.edges.items():
            key = edge_key(n)
            edges[key] = [{"to": list(nb), "cost": c, "action": list(a)} for nb, c, a in eds]

        ramp_info = {}
        for k, v in self.ramp_info.items():
            ramp_info[ramp_key(k)] = {"down": HEADING_NAMES[v[0]], "up": HEADING_NAMES[v[1]]}

        data = {
            "nodes": [list(n) for n in sorted(self.nodes)],
            "edges": edges,
            "blocked": {edge_key(k): v for k, v in self.blocked.items()},
            "ramp_info": ramp_info,
            "stats": {
                "total_nodes": len(self.nodes),
                "total_edges": sum(len(v) for v in self.edges.values()),
                "l0": sum(1 for n in self.nodes if n[2] == 0),
                "l1": sum(1 for n in self.nodes if n[2] == 1),
            },
        }
        with open(os.path.join(out_dir, "graph.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def print_graph_stats(self):
        l0 = sum(1 for n in self.nodes if n[2] == 0)
        l1 = sum(1 for n in self.nodes if n[2] == 1)
        inv = sum(1 for v in self.blocked.values() if v == "invalid_ramp")
        print(f"Граф: L0={l0}, L1={l1}, невалидных пандусов={inv}")


MAX_LABEL = 75

def print_route(cmds, cost, raw, start_h=HEADING_N):
    for i, c in enumerate(cmds, 1): print(f"   {i}. {c}")

def commands_to_short(cmds: List[str]) -> str:
    return " -> ".join(cmds) if cmds else ""


SZ = 600
CELL = 60
OFF = (SZ - 8 * CELL) // 2
TOP = 40

def _cell_center(r, c):
    return OFF + c * CELL + CELL // 2, TOP + r * CELL + CELL // 2

def _cell_rect(r, c):
    return OFF + c * CELL, TOP + r * CELL

class InteractiveRouter:
    def __init__(self, map_data: List[Dict], prefer_straight=False):
        self.router = FieldRouter(map_data)
        self.prefer_straight = prefer_straight
        self.rows, self.cols = 8, 8
        self.start = self.goal = self.last = None
        self.h = HEADING_N
        self.msg = "ЛКМ - старт/цель | ПКМ - сброс | WASD - heading | R - reset | Q/Esc - выход"
        self.short = ""
        self._make_canvas()
        cv2.namedWindow("Router", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Router", self._on_click)
        cv2.resizeWindow("Router", SZ, SZ)

    def _make_canvas(self):
        self.base = np.ones((SZ, SZ, 3), dtype=np.uint8) * 240
        for r in range(8):
            for c in range(8):
                x1, y1 = _cell_rect(r, c)
                d = next((x for x in self.router.map_data if x["row"] == r and x["col"] == c), {})
                lvl = d.get("level", 0)
                has_ramp = d.get("ramp", 0) > 0
                bg = (60, 60, 60) if (lvl == 1 or has_ramp) else (220, 220, 220)
                cv2.rectangle(self.base, (x1, y1), (x1 + CELL, y1 + CELL), bg, -1)
                cv2.rectangle(self.base, (x1, y1), (x1 + CELL, y1 + CELL), (0, 0, 0), 1)
                tc = (255, 255, 255) if (lvl == 1 or has_ramp) else (0, 0, 0)
                cv2.putText(self.base, str(r * 8 + c), (x1 + 3, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, tc, 1)
                if has_ramp:
                    cx, cy = _cell_center(r, c)
                    rd = d.get("ramp_dir_precise", 0)
                    col = (0, 165, 255)
                    al = CELL // 2 - 4
                    if rd == "S": cv2.arrowedLine(self.base, (cx, cy - al), (cx, cy + al), col, 2, line_type=cv2.LINE_AA, tipLength=0.3)
                    elif rd == "N": cv2.arrowedLine(self.base, (cx, cy + al), (cx, cy - al), col, 2, line_type=cv2.LINE_AA, tipLength=0.3)
                    elif rd == "E": cv2.arrowedLine(self.base, (cx - al, cy), (cx + al, cy), col, 2, line_type=cv2.LINE_AA, tipLength=0.3)
                    elif rd == "W": cv2.arrowedLine(self.base, (cx + al, cy), (cx - al, cy), col, 2, line_type=cv2.LINE_AA, tipLength=0.3)

    def _xy_to_cell(self, x, y):
        c = (x - OFF) // CELL
        r = (y - TOP) // CELL
        return (r, c) if 0 <= r < 8 and 0 <= c < 8 else None

    def _draw_edges(self, canvas):
        for n, eds in self.router.edges.items():
            r, cc, _ = n
            p1 = _cell_center(r, cc)
            for nb, _, act in eds:
                nr, nc, _ = nb
                p2 = _cell_center(nr, nc)
                col = (180, 50, 180) if act[0].startswith("ramp") else (80, 120, 200)
                cv2.line(canvas, p1, p2, col, 1)

    def _on_click(self, e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            cell = self._xy_to_cell(x, y)
            if not cell: return
            if not self.start:
                self.start = cell
                self.msg = f"Start={cell}. Выберите цель. "
                self.short = " "
            else:
                self.goal = cell
                res = self.router.find_path(self.start, self.goal, self.h, prefer_straight=self.prefer_straight)
                self.last = res
                if res:
                    print_route(*res, self.h)
                    c_str = f"{res[1]:.1f}"
                    self.short = commands_to_short(res[0])
                else:
                    c_str = "None"; self.short = "NO PATH"
                self.msg = f"Start={self.start}, Goal={self.goal} | Cost={c_str}"
            self._redraw()
        elif e == cv2.EVENT_RBUTTONDOWN:
            self.start = self.goal = self.last = None
            self.msg = "ЛКМ - старт/цель | ПКМ - сброс | WASD - heading | R - reset | Q/Esc - выход"
            self.short = " "
            self._redraw()

    def _redraw(self):
        c = self.base.copy()
        self._draw_edges(c)
        if self.last:
            pts = [_cell_center(r, cc) for r, cc, _ in self.last[2]]
            for i in range(len(pts) - 1):
                cv2.line(c, pts[i], pts[i + 1], (0, 120, 255), 2)
            if pts:
                cv2.circle(c, pts[0], 6, (0, 255, 0), -1)
                cv2.circle(c, pts[-1], 6, (0, 0, 255), -1)
        if self.start:
            r, cc = self.start
            x, y = _cell_rect(r, cc)
            cv2.rectangle(c, (x + 1, y + 1), (x + CELL - 2, y + CELL - 2), (0, 255, 0), 2)
            d = HEADING_DELTA[self.h]
            cx, cy = _cell_center(r, cc)
            cv2.arrowedLine(c, (cx, cy), (cx + d[1] * 20, cy + d[0] * 20), (0, 255, 0), 2, line_type=cv2.LINE_AA, tipLength=0.3)
        cv2.rectangle(c, (0, SZ - 32), (SZ, SZ), (40, 40, 40), -1)
        cv2.putText(c, self.msg, (10, SZ - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        if self.short:
            lbl = f"Route: {self.short}"
            if len(lbl) > 60:
                lbl = lbl[:57] + "..."
            cv2.putText(c, lbl, (OFF, TOP - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 200), 1)
        cv2.imshow("Router", c)

    def run(self):
        mode = "Интерактивный режим"
        if self.prefer_straight:
            mode += " [SPEED]"
        print(f"{mode}. ЛКМ - старт/цель, ПКМ - сброс, WASD - heading, Q/Esc - выход")
        self._redraw()
        while True:
            k = cv2.waitKey(50) & 0xFF
            if k in (27, ord("q")): break
            if k == ord("r"):
                self.start = self.goal = self.last = None
                self.msg = "ЛКМ - старт/цель | ПКМ - сброс | WASD - heading | R - reset | Q/Esc - выход"
                self.short = " "; self._redraw()
            elif k == ord("w"): self.h = HEADING_N; self._redraw()
            elif k == ord("a"): self.h = HEADING_W; self._redraw()
            elif k == ord("s"): self.h = HEADING_S; self._redraw()
            elif k == ord("d"): self.h = HEADING_E; self._redraw()
        cv2.destroyAllWindows()


class CarryInteractiveRouter:
    def __init__(self, map_data: List[Dict], animate=False, robot_start=None, prefer_straight=False, obj_cmd="K"):
        self.router = FieldRouter(map_data)
        from carry_planner import CarryPlanner, commands_to_short_carry
        self.planner = CarryPlanner(self.router, obj_cmd=obj_cmd)
        self.cmd_short = commands_to_short_carry
        self.prefer_straight = prefer_straight
        self.rows, self.cols = 8, 8
        self.start = None
        self.h = HEADING_N
        self.plan = None
        self.anim = animate
        self.robot_start = robot_start
        if robot_start:
            self.start = robot_start
            self.msg = f"Робот найден на {robot_start} | WASD - heading | Enter - отправить | Q - выход"
        else:
            self.msg = "ЛКМ - старт робота | WASD - heading | Enter - отправить | Q - выход"
        self.short = " "
        self.tubesPoints = []
        self.postsPoints = []
        self._make_canvas()
        cv2.namedWindow("Carry Planner", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Carry Planner", self._on_click)
        cv2.resizeWindow("Carry Planner", SZ, SZ)

    def _make_canvas(self):
        self.base = np.ones((SZ, SZ, 3), dtype=np.uint8) * 240
        for r in range(8):
            for c in range(8):
                x1, y1 = _cell_rect(r, c)
                d = next((x for x in self.router.map_data if x["row"] == r and x["col"] == c), {})
                lvl = d.get("level", 0)
                has_ramp = d.get("ramp", 0) > 0
                bg = (60, 60, 60) if (lvl == 1 or has_ramp) else (220, 220, 220)
                cv2.rectangle(self.base, (x1, y1), (x1 + CELL, y1 + CELL), bg, -1)
                cv2.rectangle(self.base, (x1, y1), (x1 + CELL, y1 + CELL), (0, 0, 0), 1)
                tc = (255, 255, 255) if (lvl == 1 or has_ramp) else (0, 0, 0)
                cv2.putText(self.base, str(r * 8 + c), (x1 + 3, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, tc, 1)
                cx, cy = _cell_center(r, c)
                if has_ramp:
                    col = (0, 165, 255)
                    rd = d.get("ramp_dir_precise", 0)
                    al = CELL // 2 - 4
                    if rd == "S": cv2.arrowedLine(self.base, (cx, cy - al), (cx, cy + al), col, 2, line_type=cv2.LINE_AA, tipLength=0.3)
                    elif rd == "N": cv2.arrowedLine(self.base, (cx, cy + al), (cx, cy - al), col, 2, line_type=cv2.LINE_AA, tipLength=0.3)
                    elif rd == "E": cv2.arrowedLine(self.base, (cx - al, cy), (cx + al, cy), col, 2, line_type=cv2.LINE_AA, tipLength=0.3)
                    elif rd == "W": cv2.arrowedLine(self.base, (cx + al, cy), (cx - al, cy), col, 2, line_type=cv2.LINE_AA, tipLength=0.3)
                rt = d.get("redTube", 0)
                if rt > 0:
                    if rt == 1: cv2.rectangle(self.base, (cx - 4, cy - 14), (cx + 4, cy + 14), (0, 0, 255), 2)
                    else: cv2.rectangle(self.base, (cx - 14, cy - 4), (cx + 14, cy + 4), (0, 0, 255), 2)
                bt = d.get("blueTube", 0)
                if bt > 0:
                    ox = 10 if rt > 0 else 0; oy = 10 if rt > 0 else 0
                    if bt == 1: cv2.rectangle(self.base, (cx - 4 + ox, cy - 14 + oy), (cx + 4 + ox, cy + 14 + oy), (255, 0, 0), 2)
                    else: cv2.rectangle(self.base, (cx - 14 + ox, cy - 4 + oy), (cx + 14 + ox, cy + 4 + oy), (255, 0, 0), 2)
                g = d.get("green", 0)
                if g > 0: cv2.circle(self.base, (cx + 16, cy + 16), 7, (0, 200, 0), 2)
                robot = d.get("robot", 0)
                if robot > 0:
                    L, Wt = 16, 10
                    pts = np.array([[cx, cy - L], [cx - Wt, cy + L // 2], [cx + Wt, cy + L // 2]], dtype=np.int32)
                    cv2.fillPoly(self.base, [pts], (0, 140, 255))
                    cv2.polylines(self.base, [pts], True, (0, 0, 0), 1)

    def _xy_to_cell(self, x, y):
        c = (x - OFF) // CELL
        r = (y - TOP) // CELL
        return (r, c) if 0 <= r < 8 and 0 <= c < 8 else None

    def _plan_from(self, cell):
        self.start = cell
        self.msg = f"Планирую от {cell} (heading={HEADING_NAMES[self.h]})... "
        self.short = "PLANNING... "
        self._redraw(); cv2.waitKey(10)
        self.plan = self.planner.plan(cell, self.h, verbose=False, animate=False, robot_start=self.robot_start, prefer_straight=self.prefer_straight)
        if self.plan:
            self.planner.print_plan(self.plan)
            self.short = self.cmd_short(self.plan["commands"])
            src = "АВТО" if self.robot_start else "РУЧНОЙ"
            self.msg = f"План готов ({src}) | Start={cell} | Cost={self.plan['total_cost']:.2f} | Сегментов: {len(self.plan['segments'])} | Enter - отправить | WASD - heading"
            if self.anim:
                self.plan["start_heading"] = self.h
                self.planner.animate_plan(self.plan)
        else:
            self.short = "NO PLAN"
            self.msg = f"План не найден от {cell} | WASD - сменить heading, R - reset"
        self._redraw()

    def _on_click(self, e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            if self.robot_start is not None: return
            cell = self._xy_to_cell(x, y)
            if not cell: return
            self._plan_from(cell)
        elif e == cv2.EVENT_RBUTTONDOWN:
            self.start = self.plan = None
            if self.robot_start:
                self.start = self.robot_start
                self.msg = f"Робот на {self.robot_start} | WASD - heading | R - reset | Q/Esc - выход"
            else:
                self.msg = "ЛКМ - старт робота | WASD - heading | R - reset | Q/Esc - выход"
            self.short = " "; self._redraw()

    def _redraw(self):
        c = self.base.copy()
        if self.plan and self.plan.get("raw_path"):
            pts = [_cell_center(r, cc) for (r, cc) in self.plan["raw_path"]]
            for i in range(len(pts) - 1):
                cv2.line(c, pts[i], pts[i + 1], (0, 120, 255), 2)
            if pts:
                cv2.circle(c, pts[0], 6, (0, 255, 0), -1)
                cv2.circle(c, pts[-1], 6, (0, 0, 255), -1)
            for seg in self.plan["segments"]:
                ta = seg["tube_approach"]; pa = seg["pod_approach"]
                tx, ty = _cell_center(ta[0], ta[1])
                px, py = _cell_center(pa[0], pa[1])
                cv2.circle(c, (tx, ty), 5, (0, 165, 255), -1)
                cv2.circle(c, (px, py), 5, (0, 255, 0), -1)
        if self.start:
            r, cc = self.start
            x, y = _cell_rect(r, cc)
            cv2.rectangle(c, (x + 1, y + 1), (x + CELL - 2, y + CELL - 2), (0, 255, 0), 2)
            d = HEADING_DELTA[self.h]
            cx, cy = _cell_center(r, cc)
            cv2.arrowedLine(c, (cx, cy), (cx + d[1] * 20, cy + d[0] * 20), (0, 255, 0), 2, line_type=cv2.LINE_AA, tipLength=0.3)
        cv2.rectangle(c, (0, SZ - 32), (SZ, SZ), (40, 40, 40), -1)
        cv2.putText(c, self.msg, (10, SZ - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        if self.short:
            lbl = f"Plan: {self.short}"
            if len(lbl) > 60:
                lbl = lbl[:57] + "..."
            cv2.putText(c, lbl, (OFF, TOP - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 200), 1)
        cv2.imshow("Carry Planner", c)

    def run(self):
        if not self.robot_start:
            print("Робот не обнаружен на поле — выберите стартовую клетку кликом мыши.")
        if not self.robot_start: print("   ЛКМ — выбрать стартовую клетку")
        self._redraw()
        if self.robot_start:
            cv2.waitKey(200)
            self._plan_from(self.robot_start)
        send = True
        while True:
            k = cv2.waitKey(50) & 0xFF
            if k in (13, 10): send = True; break
            if k in (27, ord("q")): send = False; self.plan = None; print("Отменено пользователем — отправка на робота пропущена."); break
            if k == ord("r"):
                self.plan = None
                if self.robot_start:
                    self.start = self.robot_start
                    self.msg = f"Робот на {self.robot_start} | WASD - heading | Q/Esc - выход"
                else:
                    self.start = None
                    self.msg = "ЛКМ - старт робота | WASD - heading | R - reset | Q/Esc - выход"
                self.short = " "; self._redraw()
            elif k in (ord("w"), ord("a"), ord("s"), ord("d")):
                new_h = {ord("w"): HEADING_N, ord("a"): HEADING_W, ord("s"): HEADING_S, ord("d"): HEADING_E}[k]
                self.h = new_h
                if self.robot_start: self._plan_from(self.robot_start)
                elif self.start and not self.robot_start: self._plan_from(self.start)
                else: self._redraw()
        cv2.destroyAllWindows()

        return self.plan if send else None


def run_interactive(map_data: List[Dict], prefer_straight=False):
    InteractiveRouter(map_data, prefer_straight=prefer_straight).run()

def run_carry_interactive(map_data: List[Dict], animate=False, robot_start=None, prefer_straight=False, obj_cmd="K"):
    return CarryInteractiveRouter(map_data, animate=animate, robot_start=robot_start, prefer_straight=prefer_straight, obj_cmd=obj_cmd).run()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="output/map.json")
    ap.add_argument("--start", default=None)
    ap.add_argument("--goal", default=None)
    ap.add_argument("--heading", type=int, default=0)
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--carry", action="store_true")
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--speed", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.map):
        print(f"Файл не найден: {args.map}"); return
    with open(args.map, "r", encoding="utf-8") as f: map_data = json.load(f)

    router = FieldRouter(map_data, debug=True)
    router.print_graph_stats()
    router.save_graph(os.path.dirname(args.map) or "output")

    if args.carry: run_carry_interactive(map_data, animate=args.animate, prefer_straight=args.speed); return
    if args.interactive: run_interactive(map_data, prefer_straight=args.speed); return

    if args.start and args.goal:
        start = tuple(int(x) for x in args.start.split(","))
        goal = tuple(int(x) for x in args.goal.split(","))
        res = router.find_path(start, goal, args.heading, prefer_straight=args.speed)
        if res: print_route(*res, args.heading); print(f"\nShort: {commands_to_short(res[0])}")
        else: print("Путь не найден")

if __name__ == "__main__":
    main()