# router.py
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
RAMP_COST = 1.5
MOVE_COST = 1.0

class FieldRouter:
    def __init__(self, map_data: List[Dict], debug: bool = False):
        self.map_data = map_data
        self.grid = {(item['row'], item['col']): item for item in map_data}
        self.nodes, self.edges, self.blocked, self.ramp_info = set(), {}, {}, {}
        self.debug = debug
        self._build_graph()

    def _is_blocked(self, r: int, c: int, level: int) -> Tuple[bool, str]:
        data = self.grid.get((r, c))
        if not data:
            return True, "outside"
        if data.get("redTube", 0) > 0 or data.get("blueTube", 0) > 0 or data.get("green", 0) > 0:
            return True, "obstacle"
        if data.get("ramp", 0) > 0:
            return False, ""
        return (level != data.get("level", 0)), "level_mismatch"

    def _is_valid_ramp(self, r: int, c: int, ramp_dir_precise) -> bool:
        if ramp_dir_precise not in ('N', 'S', 'E', 'W'):
            return False
            
        heading_map = {'N': HEADING_N, 'E': HEADING_E, 'S': HEADING_S, 'W': HEADING_W}
        down_h = heading_map[ramp_dir_precise]
        up_h = (down_h + 2) % 4
        
        dr_d, dc_d = HEADING_DELTA[down_h]
        r_d, c_d = r + dr_d, c + dc_d
        down_data = self.grid.get((r_d, c_d))
        
        dr_u, dc_u = HEADING_DELTA[up_h]
        r_u, c_u = r + dr_u, c + dc_u
        up_data = self.grid.get((r_u, c_u))
        
        if not down_data or not up_data:
            return False
            
        if down_data.get("level", 0) != 0 or down_data.get("ramp", 0) > 0:
            return False
            
        if up_data.get("level", 0) != 1 or up_data.get("ramp", 0) > 0:
            return False
            
        return True

    def _build_graph(self):
        potential_nodes = set()
        for (r, c), data in self.grid.items():
            has_ramp = data.get("ramp", 0) > 0
            if has_ramp:
                ramp_dir = data.get("ramp_dir_precise")
                if not self._is_valid_ramp(r, c, ramp_dir):
                    self.blocked[(r, c, 1)] = "invalid_ramp"
                    self.blocked[(r, c, 0)] = "invalid_ramp"
                    continue
                
                heading_map = {'N': HEADING_N, 'E': HEADING_E, 'S': HEADING_S, 'W': HEADING_W}
                down_h = heading_map[ramp_dir]
                up_h = (down_h + 2) % 4
                self.ramp_info[(r, c)] = (down_h, up_h)
                
                potential_nodes.add((r, c, 1))
                self.blocked[(r, c, 0)] = "under_ramp"
            else:
                level = data.get("level", 0)
                node = (r, c, level)
                blocked, reason = self._is_blocked(r, c, level)
                if not blocked:
                    potential_nodes.add(node)
                else:
                    self.blocked[node] = reason

        self.nodes = set()
        for node in potential_nodes:
            self.nodes.add(node)

        for node in self.nodes:
            self.edges[node] = []
            r, c, level = node
            data = self.grid.get((r, c))
            has_ramp = data.get("ramp", 0) > 0

            if has_ramp and level == 1:
                down_h, up_h = self.ramp_info.get((r, c), (None, None))
                if down_h is None: 
                    continue
                
                dr_d, dc_d = HEADING_DELTA[down_h]
                neighbor_down = (r + dr_d, c + dc_d, 0)
                if neighbor_down in self.nodes:
                    self.edges[node].append((neighbor_down, RAMP_COST, ("ramp_down", down_h)))
                    
                dr_u, dc_u = HEADING_DELTA[up_h]
                neighbor_up_l1 = (r + dr_u, c + dc_u, 1)
                if neighbor_up_l1 in self.nodes:
                    self.edges[node].append((neighbor_up_l1, MOVE_COST, ("move", up_h)))
                    
            elif level == 0:
                for heading, (dr, dc) in HEADING_DELTA.items():
                    nr, nc = r + dr, c + dc
                    neighbor = (nr, nc, 0)
                    if neighbor in self.nodes:
                        self.edges[node].append((neighbor, MOVE_COST, ("move", heading)))
                    
                    n_data = self.grid.get((nr, nc))
                    if n_data and n_data.get("ramp", 0) > 0:
                        info = self.ramp_info.get((nr, nc))
                        if info:
                            down_h, up_h = info
                            if heading == up_h:
                                neighbor_ramp = (nr, nc, 1)
                                if neighbor_ramp in self.nodes:
                                    self.edges[node].append((neighbor_ramp, RAMP_COST, ("ramp_up", heading)))
                                    
            elif level == 1:
                for heading, (dr, dc) in HEADING_DELTA.items():
                    nr, nc = r + dr, c + dc
                    n_data = self.grid.get((nr, nc))
                    
                    if n_data and n_data.get("ramp", 0) > 0:
                        info = self.ramp_info.get((nr, nc))
                        if info:
                            down_h, up_h = info
                            if heading == down_h:
                                neighbor_ramp = (nr, nc, 1)
                                if neighbor_ramp in self.nodes:
                                    self.edges[node].append((neighbor_ramp, MOVE_COST, ("move", heading)))
                            continue 
                    
                    neighbor = (nr, nc, 1)
                    if neighbor in self.nodes:
                        self.edges[node].append((neighbor, MOVE_COST, ("move", heading)))

    def _heuristic(self, a: Tuple, b: Tuple) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]) * 2

    def find_path(self, start, goal, start_heading=HEADING_N, goal_heading=None, prefer_straight: bool = False):
        start_data, goal_data = self.grid.get(start), self.grid.get(goal)
        if not start_data or not goal_data:
            return None
        start_level = 1 if start_data.get("ramp", 0) > 0 else start_data.get("level", 0)
        goal_level = 1 if goal_data.get("ramp", 0) > 0 else goal_data.get("level", 0)
        start_node = (start[0], start[1], start_level)
        goal_node = (goal[0], goal[1], goal_level)

        if start_node not in self.nodes or goal_node not in self.nodes:
            return None

        counter = 0
        open_set = []
        start_state = (*start_node, start_heading)

        if prefer_straight:
            heapq.heappush(open_set, (0.0, 0, 0.0, counter, start_state))
            g_score = {start_state: (0.0, 0)}
        else:
            heapq.heappush(open_set, (0.0, 0.0, counter, start_state))
            g_score = {start_state: 0.0}

        came_from = {}

        while open_set:
            if prefer_straight:
                _, cur_turns, g, _, current = heapq.heappop(open_set)
                if (g, cur_turns) > g_score.get(current, (float('inf'), float('inf'))):
                    continue
            else:
                _, g, _, current = heapq.heappop(open_set)
                if g > g_score.get(current, float('inf')):
                    continue

            current_node = current[:3]
            current_heading = current[3]

            if current_node == goal_node:
                if goal_heading is None or current_heading == goal_heading:
                    total_cost = g_score[current][0] if prefer_straight else g_score[current]
                    return self._reconstruct_path(came_from, current, start_state, total_cost)

            for neighbor_node, cost, action in self.edges.get(current_node, []):
                action_type, action_heading = action
                neighbor_state = (*neighbor_node, action_heading)
                tentative_g = g + cost

                if prefer_straight:
                    tentative_turns = cur_turns + (1 if action_heading != current_heading else 0)
                    if (tentative_g, tentative_turns) < g_score.get(neighbor_state, (float('inf'), float('inf'))):
                        came_from[neighbor_state] = (current, action, cost)
                        g_score[neighbor_state] = (tentative_g, tentative_turns)
                        h = self._heuristic(neighbor_node, goal_node)
                        counter += 1
                        heapq.heappush(open_set, (tentative_g + h, tentative_turns, tentative_g, counter, neighbor_state))
                else:
                    if tentative_g < g_score.get(neighbor_state, float('inf')):
                        came_from[neighbor_state] = (current, action, cost)
                        g_score[neighbor_state] = tentative_g
                        h = self._heuristic(neighbor_node, goal_node)
                        counter += 1
                        heapq.heappush(open_set, (tentative_g + h, tentative_g, counter, neighbor_state))
        return None

    def _reconstruct_path(self, came_from, end_state, start_state, total_cost):
        states, actions = [end_state], []
        current = end_state
        while current != start_state:
            prev, action, _ = came_from[current]
            actions.append(action)
            states.append(prev)
            current = prev
        states.reverse()
        actions.reverse()
        raw_path = [s[:3] for s in states]
        commands = self._actions_to_commands(states, actions)
        return commands, total_cost, raw_path

    def _actions_to_commands(self, states, actions) -> List[str]:
        if not actions:
            return []
        commands = []
        current_heading = states[0][3]
        i = 0
        while i < len(actions):
            action_type, action_heading = actions[i]
            if action_heading != current_heading:
                diff = (action_heading - current_heading) % 4
                if diff == 1: commands.append("R")
                elif diff == 3: commands.append("L")
                elif diff == 2: commands.append("around")
                current_heading = action_heading
            if action_type == "move":
                n = 1
                i += 1
                while i < len(actions) and actions[i] == ("move", current_heading):
                    n += 1; i += 1
                commands.append(f"F{n}")
            elif action_type == "ramp_up":
                commands.append("U")
                current_heading = action_heading
                i += 1
            elif action_type == "ramp_down":
                commands.append("D")
                current_heading = action_heading
                i += 1
        return commands

    def save_graph(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        graph_data = {
            "nodes": [list(n) for n in sorted(self.nodes)],
            "edges": {f"{n[0]},{n[1]},{n[2]}": [{"to": list(nb), "cost": c, "action": list(a)} for nb, c, a in edges] for n, edges in self.edges.items()},
            "blocked": {f"{k[0]},{k[1]},{k[2]}": v for k, v in self.blocked.items()},
            "ramp_info": {f"{k[0]},{k[1]}": {"down": HEADING_NAMES[v[0]], "up": HEADING_NAMES[v[1]]} for k, v in self.ramp_info.items()},
            "stats": {"total_nodes": len(self.nodes), "total_edges": sum(len(v) for v in self.edges.values()), "level_0_nodes": sum(1 for n in self.nodes if n[2] == 0), "level_1_nodes": sum(1 for n in self.nodes if n[2] == 1)},
        }
        with open(os.path.join(out_dir, "graph.json"), "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)

    def print_graph_stats(self):
        l0 = sum(1 for n in self.nodes if n[2] == 0)
        l1 = sum(1 for n in self.nodes if n[2] == 1)
        invalid_ramps = sum(1 for v in self.blocked.values() if v == "invalid_ramp")
        print(f"Граф построен: узлов L0={l0}, L1={l1}, невалидных пандусов={invalid_ramps}")

def print_route(commands, total_cost, raw_path, start_heading=HEADING_N):
    for i, cmd in enumerate(commands, 1): print(f"   {i}. {cmd}")

def commands_to_short(commands: List[str]) -> str:
    return " -> ".join(commands) if commands else ""

class InteractiveRouter:
    def __init__(self, map_data: List[Dict], cell_size: int = 100, prefer_straight: bool = False):
        self.router = FieldRouter(map_data)
        self.prefer_straight = prefer_straight
        self.cell_size, self.margin = cell_size, 80
        self.rows, self.cols = 8, 8
        self.start_cell = self.goal_cell = self.last_path = None
        self.start_heading = HEADING_N
        self.status_msg = "ЛКМ - старт/цель | ПКМ - сброс | WASD - heading | R - reset | Q/Esc - выход"
        self.commands_short = ""
        self._build_canvas()
        cv2.namedWindow("Router", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Router", self._on_mouse)

    def _build_canvas(self):
        cs, m = self.cell_size, self.margin
        h = self.rows * cs + m * 2 + 120
        w = self.cols * cs + m * 2
        self.canvas_base = np.ones((h, w, 3), dtype=np.uint8) * 240
        for r in range(self.rows):
            for c in range(self.cols):
                x1, y1 = m + c * cs, m + r * cs
                data = next((d for d in self.router.map_data if d["row"] == r and d["col"] == c), {})
                lvl = data.get("level", 0)
                has_ramp = data.get("ramp", 0) > 0
                bg = (60, 60, 60) if (lvl == 1 or has_ramp) else (200, 200, 200)
                cv2.rectangle(self.canvas_base, (x1, y1), (x1 + cs, y1 + cs), bg, -1)
                cv2.rectangle(self.canvas_base, (x1, y1), (x1 + cs, y1 + cs), (0, 0, 0), 2)
                txt_col = (255, 255, 255) if (lvl == 1 or has_ramp) else (0, 0, 0)
                cv2.putText(self.canvas_base, str(r * 8 + c), (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_col, 2)
                if has_ramp:
                    cx, cy = x1 + cs // 2, y1 + cs // 2
                    col = (0, 165, 255)
                    ramp_dir = data.get("ramp_dir_precise", 0)
                    if ramp_dir == 'S': cv2.arrowedLine(self.canvas_base, (cx, cy - 30), (cx, cy + 30), col, 4, line_type=cv2.LINE_AA, tipLength=0.3)
                    elif ramp_dir == 'N': cv2.arrowedLine(self.canvas_base, (cx, cy + 30), (cx, cy - 30), col, 4, line_type=cv2.LINE_AA, tipLength=0.3)
                    elif ramp_dir == 'E': cv2.arrowedLine(self.canvas_base, (cx - 30, cy), (cx + 30, cy), col, 4, line_type=cv2.LINE_AA, tipLength=0.3)
                    elif ramp_dir == 'W': cv2.arrowedLine(self.canvas_base, (cx + 30, cy), (cx - 30, cy), col, 4, line_type=cv2.LINE_AA, tipLength=0.3)

    def _xy_to_cell(self, x, y):
        c = (x - self.margin) // self.cell_size
        r = (y - self.margin) // self.cell_size
        return (r, c) if 0 <= r < 8 and 0 <= c < 8 else None

    def _draw_edges(self, canvas):
        cs, m = self.cell_size, self.margin
        for node, edges in self.router.edges.items():
            r, c, _ = node
            p1 = (m + c * cs + cs // 2, m + r * cs + cs // 2)
            for nb, cost, action in edges:
                nr, nc, _ = nb
                p2 = (m + nc * cs + cs // 2, m + nr * cs + cs // 2)
                col = (180, 50, 180) if action[0].startswith("ramp") else (80, 120, 200)
                cv2.line(canvas, p1, p2, col, 2)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cell = self._xy_to_cell(x, y)
            if not cell: return
            if not self.start_cell:
                self.start_cell = cell
                self.status_msg = f"Start={cell}. Выберите цель."
                self.commands_short = ""
            else:
                self.goal_cell = cell
                res = self.router.find_path(self.start_cell, self.goal_cell, self.start_heading, prefer_straight=self.prefer_straight) 
                self.last_path = res
                if res:
                    print_route(*res, self.start_heading)
                    cost_str = f"{res[1]:.1f}"
                    self.commands_short = commands_to_short(res[0])
                else:
                    cost_str = "None"; self.commands_short = "NO PATH"
                self.status_msg = f"Start={self.start_cell}, Goal={self.goal_cell} | Cost={cost_str}"
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.start_cell = self.goal_cell = self.last_path = None
            self.status_msg = "ЛКМ - старт/цель | ПКМ - сброс | WASD - heading | R - reset | Q/Esc - выход"
            self.commands_short = ""
            self._redraw()

    def _redraw(self):
        canvas = self.canvas_base.copy()
        self._draw_edges(canvas)
        if self.last_path:
            pts = [(self.margin + c * self.cell_size + self.cell_size // 2, self.margin + r * self.cell_size + self.cell_size // 2) for r, c, _ in self.last_path[2]]
            for i in range(len(pts) - 1): cv2.line(canvas, pts[i], pts[i + 1], (0, 120, 255), 3)
            cv2.circle(canvas, pts[0], 8, (0, 255, 0), -1); cv2.circle(canvas, pts[-1], 8, (0, 0, 255), -1)
        if self.start_cell:
            r, c = self.start_cell
            x = self.margin + c * self.cell_size; y = self.margin + r * self.cell_size
            cv2.rectangle(canvas, (x + 2, y + 2), (x + self.cell_size - 2, y + self.cell_size - 2), (0, 255, 0), 3)
            d = HEADING_DELTA[self.start_heading]
            cx = x + self.cell_size // 2; cy = y + self.cell_size // 2
            cv2.arrowedLine(canvas, (cx, cy), (cx + d[1] * 30, cy + d[0] * 30), (0, 255, 0), 4, line_type=cv2.LINE_AA, tipLength=0.3)
        bar_h = 80; h_can, w_can = canvas.shape[:2]
        cv2.rectangle(canvas, (0, h_can - bar_h), (w_can, h_can), (40, 40, 40), -1)
        cv2.putText(canvas, self.status_msg, (self.margin, h_can - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if self.commands_short: 
            label = f"Route: {self.commands_short}"
            if len(label) > 75: label = label[:72] + "..."
            cv2.putText(canvas, label, (self.margin, h_can - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1)
        cv2.imshow("Router", canvas)

    def run(self):
        mode = "Интерактивный режим"
        if self.prefer_straight: mode += " [SPEED: меньше поворотов]"
        print(f"{mode}. ЛКМ - старт/цель, ПКМ - сброс, WASD - heading, Q/Esc - выход")
        self._redraw()
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (27, ord('q')): break
            if key == ord('r'):
                self.start_cell = self.goal_cell = self.last_path = None
                self.status_msg = "ЛКМ - старт/цель | ПКМ - сброс | WASD - heading | R - reset | Q/Esc - выход"
                self.commands_short = ""
                self._redraw()
            elif key == ord('w'): self.start_heading = HEADING_N; self._redraw()
            elif key == ord('a'): self.start_heading = HEADING_W; self._redraw()
            elif key == ord('s'): self.start_heading = HEADING_S; self._redraw()
            elif key == ord('d'): self.start_heading = HEADING_E; self._redraw()
        cv2.destroyAllWindows()

class CarryInteractiveRouter:
    def __init__(self, map_data: List[Dict], cell_size: int = 100, animate: bool = False, robot_start: Optional[Tuple[int, int]] = None, prefer_straight: bool = False):
        self.router = FieldRouter(map_data)
        from carry_planner import CarryPlanner, commands_to_short_carry
        self.planner = CarryPlanner(self.router)
        self.cmd_to_short = commands_to_short_carry
        self.prefer_straight = prefer_straight
        self.cell_size, self.margin = cell_size, 80
        self.rows, self.cols = 8, 8
        self.start_cell = None
        self.start_heading = HEADING_N
        self.plan = None
        self.animate_enabled = animate
        self.robot_start = robot_start
        if robot_start:
            self.start_cell = robot_start
            self.status_msg = f"Робот найден на {robot_start} | WASD - heading | Enter - отправить | Q - выход"
        else:
            self.status_msg = "ЛКМ - старт робота | WASD - heading | Enter - отправить | Q - выход"
        self.commands_short = ""
        self._build_canvas()
        cv2.namedWindow("Carry Planner", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Carry Planner", self._on_mouse)

    def _build_canvas(self):
        cs, m = self.cell_size, self.margin
        h = self.rows * cs + m * 2 + 120
        w = self.cols * cs + m * 2
        self.canvas_base = np.ones((h, w, 3), dtype=np.uint8) * 240
        for r in range(self.rows):
            for c in range(self.cols):
                x1, y1 = m + c * cs, m + r * cs
                data = next((d for d in self.router.map_data if d["row"] == r and d["col"] == c), {})
                lvl = data.get("level", 0)
                has_ramp = data.get("ramp", 0) > 0
                bg = (60, 60, 60) if (lvl == 1 or has_ramp) else (200, 200, 200)
                cv2.rectangle(self.canvas_base, (x1, y1), (x1 + cs, y1 + cs), bg, -1)
                cv2.rectangle(self.canvas_base, (x1, y1), (x1 + cs, y1 + cs), (0, 0, 0), 2)
                txt_col = (255, 255, 255) if (lvl == 1 or has_ramp) else (0, 0, 0)
                cv2.putText(self.canvas_base, str(r * 8 + c), (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_col, 2)
                cx, cy = x1 + cs // 2, y1 + cs // 2
                if has_ramp:
                    col = (0, 165, 255)
                    ramp_dir = data.get("ramp_dir_precise", 0)
                    if ramp_dir == 'S': cv2.arrowedLine(self.canvas_base, (cx, cy - 30), (cx, cy + 30), col, 4, line_type=cv2.LINE_AA, tipLength=0.3)
                    elif ramp_dir == 'N': cv2.arrowedLine(self.canvas_base, (cx, cy + 30), (cx, cy - 30), col, 4, line_type=cv2.LINE_AA, tipLength=0.3)
                    elif ramp_dir == 'E': cv2.arrowedLine(self.canvas_base, (cx - 30, cy), (cx + 30, cy), col, 4, line_type=cv2.LINE_AA, tipLength=0.3)
                    elif ramp_dir == 'W': cv2.arrowedLine(self.canvas_base, (cx + 30, cy), (cx - 30, cy), col, 4, line_type=cv2.LINE_AA, tipLength=0.3)
                rt = data.get("redTube", 0)
                if rt > 0:
                    if rt == 1: cv2.rectangle(self.canvas_base, (cx - 6, cy - 22), (cx + 6, cy + 22), (0, 0, 255), 3)
                    else: cv2.rectangle(self.canvas_base, (cx - 22, cy - 6), (cx + 22, cy + 6), (0, 0, 255), 3)
                bt = data.get("blueTube", 0)
                if bt > 0:
                    offset_x = 14 if rt > 0 else 0; offset_y = 14 if rt > 0 else 0
                    if bt == 1: cv2.rectangle(self.canvas_base, (cx - 6 + offset_x, cy - 22 + offset_y), (cx + 6 + offset_x, cy + 22 + offset_y), (255, 0, 0), 3)
                    else: cv2.rectangle(self.canvas_base, (cx - 22 + offset_x, cy - 6 + offset_y), (cx + 22 + offset_x, cy + 6 + offset_y), (255, 0, 0), 3)
                g = data.get("green", 0)
                if g > 0: cv2.circle(self.canvas_base, (cx + 25, cy + 25), 10, (0, 200, 0), 3)
                robot = data.get("robot", 0)
                if robot > 0:
                    L, W_tri = 22, 14
                    pts = np.array([[cx, cy - L], [cx - W_tri, cy + L // 2], [cx + W_tri, cy + L // 2]], dtype=np.int32)
                    cv2.fillPoly(self.canvas_base, [pts], (0, 140, 255))
                    cv2.polylines(self.canvas_base, [pts], True, (0, 0, 0), 2)

    def _xy_to_cell(self, x, y):
        c = (x - self.margin) // self.cell_size
        r = (y - self.margin) // self.cell_size
        return (r, c) if 0 <= r < 8 and 0 <= c < 8 else None

    def _plan_from(self, cell: Tuple[int, int]):
        self.start_cell = cell
        self.status_msg = f"Планирую от {cell} (heading={HEADING_NAMES[self.start_heading]})..."
        self.commands_short = "PLANNING..."
        self._redraw(); cv2.waitKey(10)
        self.plan = self.planner.plan(cell, self.start_heading, verbose=True, animate=False, robot_start=self.robot_start, prefer_straight=self.prefer_straight)
        if self.plan:
            self.planner.print_plan(self.plan)
            self.commands_short = self.cmd_to_short(self.plan["commands"])
            src = "АВТО" if self.robot_start else "РУЧНОЙ"
            self.status_msg = f"План готов ({src}) | Start={cell} | Cost={self.plan['total_cost']:.2f} | Сегментов: {len(self.plan['segments'])} | Enter - отправить | WASD - heading"
            if self.animate_enabled:
                self.plan["start_heading"] = self.start_heading
                self.planner.animate_plan(self.plan)
        else:
            self.commands_short = "NO PLAN"
            self.status_msg = f"План не найден от {cell} | WASD - сменить heading, R - reset"
        self._redraw()

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.robot_start is not None: return
            cell = self._xy_to_cell(x, y)
            if not cell: return
            self._plan_from(cell)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.start_cell = self.plan = None
            if self.robot_start:
                self.start_cell = self.robot_start
                self.status_msg = f"Робот на {self.robot_start} | WASD - heading | R - reset | Q/Esc - выход"
            else:
                self.status_msg = "ЛКМ - старт робота | WASD - heading | R - reset | Q/Esc - выход"
            self.commands_short = ""
            self._redraw()

    def _redraw(self):
        canvas = self.canvas_base.copy()
        if self.plan and self.plan.get("raw_path"):
            cs, m = self.cell_size, self.margin
            pts = [(m + c * cs + cs // 2, m + r * cs + cs // 2) for (r, c) in self.plan["raw_path"]]
            for i in range(len(pts) - 1): cv2.line(canvas, pts[i], pts[i + 1], (0, 120, 255), 3)
            if pts:
                cv2.circle(canvas, pts[0], 10, (0, 255, 0), -1)
                cv2.circle(canvas, pts[-1], 10, (0, 0, 255), -1)
            for seg in self.plan["segments"]:
                ta = seg["tube_approach"]; pa = seg["pod_approach"]
                tx = m + ta[1] * cs + cs // 2; ty = m + ta[0] * cs + cs // 2
                px = m + pa[1] * cs + cs // 2; py = m + pa[0] * cs + cs // 2
                cv2.circle(canvas, (tx, ty), 6, (0, 165, 255), -1)
                cv2.circle(canvas, (px, py), 6, (0, 255, 0), -1)
        if self.start_cell:
            r, c = self.start_cell
            x = self.margin + c * self.cell_size; y = self.margin + r * self.cell_size
            cv2.rectangle(canvas, (x + 2, y + 2), (x + self.cell_size - 2, y + self.cell_size - 2), (0, 255, 0), 3)
            d = HEADING_DELTA[self.start_heading]
            cx = x + self.cell_size // 2; cy = y + self.cell_size // 2
            cv2.arrowedLine(canvas, (cx, cy), (cx + d[1] * 30, cy + d[0] * 30), (0, 255, 0), 4, line_type=cv2.LINE_AA, tipLength=0.3)
        bar_h = 80; h_can, w_can = canvas.shape[:2]
        cv2.rectangle(canvas, (0, h_can - bar_h), (w_can, h_can), (40, 40, 40), -1)
        cv2.putText(canvas, self.status_msg, (self.margin, h_can - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if self.commands_short:
            label = f"Plan: {self.commands_short}"
            if len(label) > 75: label = label[:72] + "..."
            cv2.putText(canvas, label, (self.margin, h_can - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 1)
        cv2.imshow("Carry Planner", canvas)

    def run(self):
        if not self.robot_start:
            print("Робот не обнаружен на поле — выберите стартовую клетку кликом мыши.")
            
        if not self.robot_start: print("   ЛКМ — выбрать стартовую клетку")
        self._redraw()
        if self.robot_start:
            cv2.waitKey(200)
            self._plan_from(self.robot_start)
        send_to_robot = False
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (13, 10): send_to_robot = True; break
            if key in (27, ord('q')): send_to_robot = False; self.plan = None; print("Отменено пользователем — отправка на робота пропущена."); break
            if key == ord('r'):
                self.plan = None
                if self.robot_start:
                    self.start_cell = self.robot_start
                    self.status_msg = f"Робот на {self.robot_start} | WASD - heading | Q/Esc - выход"
                else:
                    self.start_cell = None
                    self.status_msg = "ЛКМ - старт робота | WASD - heading | R - reset | Q/Esc - выход"
                self.commands_short = ""
                self._redraw()
            elif key in (ord('w'), ord('a'), ord('s'), ord('d')):
                new_heading = {ord('w'): HEADING_N, ord('a'): HEADING_W, ord('s'): HEADING_S, ord('d'): HEADING_E}[key]
                self.start_heading = new_heading
                if self.robot_start: self._plan_from(self.robot_start)
                elif self.start_cell and not self.robot_start: self._plan_from(self.start_cell)
                else: self._redraw()
        cv2.destroyAllWindows()
        return self.plan if send_to_robot else None

def run_interactive(map_data: List[Dict], cell_size: int = 100, prefer_straight: bool = False):
    InteractiveRouter(map_data, cell_size, prefer_straight=prefer_straight).run()

def run_carry_interactive(map_data: List[Dict], cell_size: int = 100, animate: bool = False, robot_start: Optional[Tuple[int, int]] = None, prefer_straight: bool = False):
    return CarryInteractiveRouter(map_data, cell_size, animate=animate, robot_start=robot_start, prefer_straight=prefer_straight).run()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="output/map.json")
    ap.add_argument("--start", default=None)
    ap.add_argument("--goal", default=None)
    ap.add_argument("--heading", type=int, default=0)
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--carry", action="store_true", help="Режим планирования перевозки труб")
    ap.add_argument("--animate", action="store_true", help="Показать анимацию выполнения плана")
    ap.add_argument("--speed", action="store_true", help="При равной длине маршрута выбирать путь с меньшим числом поворотов")
    args = ap.parse_args()

    if not os.path.isfile(args.map): print(f"Файл не найден: {args.map}"); return
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