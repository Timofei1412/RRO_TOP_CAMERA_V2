import sys
import time
from typing import List, Optional

try:
    import serial
    import serial.tools.list_ports
    if not hasattr(serial, "Serial"):
        raise ImportError("Invalid serial package")
except ImportError:
    print("=" * 60)
    print("ERROR: Wrong 'serial' package installed.")
    print("Required: pyserial")
    print("Run: pip uninstall serial -y && pip install pyserial")
    print("=" * 60)
    sys.exit(1)


class BluetoothSender:
    def __init__(self, baudrate: int = 115200, timeout: int = 10):
        self.baudrate = baudrate
        self.timeout = timeout
        self.conn = None
        self.connected = False
        self.port = None

    def connect(self, port: str) -> bool:
        try:
            print(f"Connecting to {port}...", end=" ")
            self.conn = serial.Serial(port, self.baudrate, timeout=self.timeout)
            time.sleep(3)
            if self.conn.is_open:
                self.connected = True
                self.port = port
                print("OK")
                return True
            self.conn.close()
            return False
        except Exception as e:
            print(f"Error: {e}")
            return False

    def connect_to(self, port: Optional[str] = None) -> bool:
        return self.connect(port) if port else self.connect_any()

    def disconnect(self):
        if self.conn and self.conn.is_open:
            time.sleep(1)
            self.conn.flush()
            self.conn.close()
        self.connected = False

    @staticmethod
    def to_compact(commands: List[str], start_level: int = 0) -> str:
        prefix = "w" if start_level == 0 else "b"
        result = [prefix]
        for cmd in commands:
            c = cmd.strip().upper()
            if c.startswith("F"):
                result.append(c)
            elif c == "AROUND":
                result.append("A")
            elif c in ("R", "L", "U", "D", "T", "P","S"):
                result.append(c)
        return "".join(result)

    def send_compact(self, commands: List[str], start_level: int = 0) -> bool:
        if not self.connected:
            return False
        try:
            msg = self.to_compact(commands, start_level) + "\n"
            time.sleep(0.5)
            self.conn.write(msg.encode("utf-8"))
            self.conn.flush()
            print(f"Sent: {msg.strip()}")
            time.sleep(1.0)
            return True
        except Exception as e:
            print(f"Send error: {e}")
            return False

    def read_response(self, timeout: int = 60) -> List[str]:
        responses = []
        start = time.time()
        while time.time() - start < timeout:
            if self.conn and self.conn.in_waiting > 0:
                line = self.conn.readline().decode("utf-8").strip()
                if line:
                    responses.append(line)
                    print(f"ESP32: {line}")
                    if "DONE" in line or "ERR" in line:
                        break
            time.sleep(0.3)
        return responses


def run(commands: Optional[List[str]] = None, com_port: Optional[str] = None,
        start_level: int = 0, simulate: bool = False) -> bool:
    if simulate or not com_port:
        print(f"Simulation: {BluetoothSender.to_compact(commands or [], start_level)}")
        return True

    sender = BluetoothSender()
    if not sender.connect_to(com_port):
        print("Connection failed. Falling back to simulation.")
        return run(commands, None, start_level, simulate=True)

    if commands:
        sender.send_compact(commands, start_level)
        sender.read_response(timeout=60)

    sender.disconnect()
    print("Done")
    return True

if __name__ == "__main__":
    test = ["F1", "S", "around"]
    sender = BluetoothSender()
    if not sender.connect_to("COM5"):
        print("Connection failed. Falling back to simulation.")
    sender.send_compact(test)
    print(test)
    sender.read_response(timeout=60)

    sender.disconnect()
    print("Done")
