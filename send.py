"""
send.py — Отправка команд на ESP32 через Bluetooth.
Формат: [w|b][команды]\n
"""
import serial
import serial.tools.list_ports
import time
from typing import List, Optional


class BluetoothSender:
    def __init__(self, baudrate: int = 115200, timeout: int = 10):
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
        self.connected = False
        self.port = None

    def find_all_ports(self) -> List[str]:
        ports = serial.tools.list_ports.comports()
        for port in ports:
            print(f"Порт: {port.device} - {port.description}")
        return [p.device for p in ports]

    def connect(self, port: str) -> bool:
        try:
            print(f"   Подключение к {port}...  ", end=" ")
            self.serial_conn = serial.Serial(port, self.baudrate, timeout=self.timeout)
            time.sleep(3) 
            if self.serial_conn.is_open:
                self.connected = True
                self.port = port
                print("Успех!")
                return True
            self.serial_conn.close()
            return False
        except Exception as e:
            print(f"Ошибка: {e}")
            return False

    def connect_any(self) -> bool:
        ports = self.find_all_ports()
        for port in ports:
            if self.connect(port):
                return True
        return False

    def connect_specific(self, port: str) -> bool:
        return self.connect(port) if port else self.connect_any()

    def disconnect(self):
        if self.serial_conn and self.serial_conn.is_open:
            time.sleep(1)  
            self.serial_conn.flush()
            self.serial_conn.close()
        self.connected = False

    @staticmethod
    def commands_to_compact(commands: List[str], start_level: int = 0) -> str:
        level_char = 'w' if start_level == 0 else 'b'
        parts = [level_char]
        for cmd in commands:
            cmd = cmd.strip().upper()
            if cmd.startswith('F'):
                parts.append(cmd)
            elif cmd == 'AROUND':
                parts.append('A')
            elif cmd in ('R', 'L', 'U', 'D', 'T', 'P'):
                parts.append(cmd)
        return ''.join(parts)

    def send_compact(self, commands: List[str], start_level: int = 0) -> bool:
        if not self.connected:
            return False
        try:
            compact = self.commands_to_compact(commands, start_level)
            msg = f"{compact}\n"
            
            time.sleep(0.5)  # 👇 перед отправкой
            self.serial_conn.write(msg.encode('utf-8'))
            self.serial_conn.flush()
            print(f"Отправлено: {compact}")
            
            time.sleep(1.0)  # 👇 ждать обработки ESP32
            return True
        except Exception as e:
            print(f"Ошибка: {e}")
            return False

    def read_response(self, timeout: int = 60) -> List[str]:
        """Читает ответы от ESP32 до DONE/ERR или таймаута."""
        responses = []
        start = time.time()
        while time.time() - start < timeout:
            if self.serial_conn.in_waiting > 0:
                line = self.serial_conn.readline().decode('utf-8').strip()
                if line:
                    responses.append(line)
                    print(f"ESP32: {line}")
                    if "DONE" in line or "ERR" in line:
                        break
            time.sleep(0.3)
        return responses


def run(commands: List[str] = None, com_port: str = None, 
        start_level: int = 0, simulate: bool = False) -> bool:
    if simulate or not com_port:
        compact = BluetoothSender.commands_to_compact(commands or [], start_level)
        print(f"Симуляция: {compact}")
        return True

    sender = BluetoothSender()
    if not sender.connect_specific(com_port):
        print("Режим симуляции")
        return run(commands, None, start_level, simulate=True)

    if commands:
        sender.send_compact(commands, start_level)
        sender.read_response(timeout=60)

    sender.disconnect()
    print("✅ Готово")
    return True


if __name__ == "__main__":
    test_cmds = ["F3", "R", "F2", "T", "L", "P"]
    print(f"Тест: {BluetoothSender.commands_to_compact(test_cmds, 0)}")
    test_cmds = ["F3", "R", "F2", "T", "L", "P"]
    print(f"Тест: {BluetoothSender.commands_to_compact(test_cmds, 0)}")