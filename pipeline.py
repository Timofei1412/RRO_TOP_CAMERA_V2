import sys
import os
from typing import Tuple, Optional
import normalize
import split
import map
import router
import send
import cv2
import argparse


def run_pipeline(source: str,
                 debug: bool = False,
                 interactive: bool = False,
                 carry: bool = False,
                 com_port: str = None,
                 animate: bool = False,
                 speed: bool = False,
                 hands: bool = False) -> bool:
    out_dir = "output"
    field_size: Tuple[int, int] = (1000, 1000)
    grid: Tuple[int, int] = (8, 8)
    section_prefix = "section"
    field_filename = "field.png"

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("ЗАПУСК ПАЙПЛАЙНА RRO 2026 — Будущие инженеры")
    if carry:
        print("РЕЖИМ: Carry Planner (перевозка труб)")
    if animate:
        print("АНИМАЦИЯ: Включена")
    if speed:
        print("SPEED: при равной длине — меньше поворотов")
    if hands:
        print("HANDS: ручной выбор углов разрешён при сбое авто-поиска")
    else:
        print("HANDS: OFF — при сбое авто-поиска используются сохранённые координаты")
    print("=" * 60)

    # Шаг 1: Нормализация
    normalized_field = normalize.extract_field(
        source=source, debug=debug, out_size=field_size, hands=hands
    )
    if normalized_field is None:
        print("normalize.py не смог найти красную рамку.")
        return False

    field_path = os.path.join(out_dir, field_filename)
    cv2.imwrite(field_path, normalized_field)

    # Шаг 2: Нарезка на секции
    sections = split.run(
        normalized_field=normalized_field,
        out_dir=out_dir,
        grid=grid,
        prefix=section_prefix,
        debug=debug
    )

    # Шаг 3: Анализ секций
    map_data = map.run(sections=sections, out_dir=out_dir, debug=debug)

    robot_start: Optional[Tuple[int, int]] = None
    start_level: int = 0
    if carry:
        robot_cells = [it for it in map_data if it.get("robot", 0) > 0]
        if robot_cells:
            rc = robot_cells[0]
            robot_start = (rc["row"], rc["col"])
            start_level = rc.get("level", 0)
        else:
            print(f"\n[DETECT] Робот НЕ найден на поле — "
                  f"потребуется ручной выбор стартовой клетки.")

    # Шаг 4: Построение графа
    field_router = router.FieldRouter(map_data, debug=True)
    field_router.print_graph_stats()
    field_router.save_graph(out_dir)

    all_commands = []
    success = True

    if carry:
        plan = router.run_carry_interactive(
            map_data, animate=animate, robot_start=robot_start,
            prefer_straight=speed,
        )
        if plan:
            all_commands = plan.get("commands", [])
        success = True
    elif interactive:
        router.run_interactive(map_data, prefer_straight=speed)
        success = True
    else:
        success = True

    # Шаг 5: ОТПРАВКА ДАННЫХ
    if all_commands:
        print(f"   Отправка {len(all_commands)} команд (start_level={start_level})...")
        success = success and send.run(
            commands=all_commands,
            com_port=com_port,
            start_level=start_level,
        )
    else:
        print("Нет команд для отправки (отменено или не построено)")

    return success


def main():
    ap = argparse.ArgumentParser(
        description="Полный пайплайн RRO 2026: normalize -> split -> map -> router -> send"
    )
    ap.add_argument("source", help="Путь к файлу ИЛИ rtsp://... URL")
    ap.add_argument("-i", "--interactive", action="store_true",
                    help="Запустить интерактивный режим роутинга")
    ap.add_argument("--carry", action="store_true",
                    help="Режим планирования перевозки всех труб к подставкам")
    ap.add_argument("--animate", action="store_true",
                    help="Показать анимацию выполнения плана (только для --carry)")
    ap.add_argument("--port", type=str, default=None,
                    help="COM-порт для Bluetooth (например: COM4)")
    ap.add_argument("--speed", action="store_true",
                    help="При равной длине маршрута выбирать путь с меньшим числом поворотов")
    ap.add_argument("--hands", action="store_true",
                    help="Разрешить ручной выбор углов при сбое авто-поиска")
    ap.add_argument("-d", "--debug", action="store_true",
                    help="Режим отладки с визуализацией")
    args = ap.parse_args()

    if args.carry and args.interactive:
        print("Флаги --carry и --interactive взаимоисключающие. Использую --carry.")
        args.interactive = False

    success = run_pipeline(
        source=args.source,
        debug=args.debug,
        interactive=args.interactive,
        carry=args.carry,
        com_port=args.port,
        animate=args.animate,
        speed=args.speed,
        hands=args.hands,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()