import csv
import minimalmodbus
import serial
import time
import re
import threading
import os
from itertools import product
DEVICE_ADDRESS = 0x01
BAUD_RATE = 9600
TIMEOUT = 1
PORT = 'COM3'
IP="192.168.137.220"
PASSWORD="123456"
USERNAME="jetson"


def cmd_ssh(host, username, password=None, key_filepath=None, command="echo hello", port=22, timeout=10, use_sudo=False):
    try:
        import paramiko
        import sys
    except ImportError:
        raise RuntimeError("paramiko is required: run `pip install paramiko`")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        if key_filepath:
            pkey = paramiko.RSAKey.from_private_key_file(key_filepath)
            client.connect(hostname=host, port=port, username=username, pkey=pkey, timeout=timeout)
        else:
            client.connect(hostname=host, port=port, username=username, password=password, timeout=timeout)

        exec_cmd = command
        get_pty = False
        if use_sudo:
            exec_cmd = "sudo -S -p '' " + command
            get_pty = True

        stdin, stdout, stderr = client.exec_command(exec_cmd, get_pty=get_pty)

        # If using sudo and a password is provided, send it via stdin (note: security risk)
        if use_sudo and password:
            try:
                stdin.write(password + "\n")
                stdin.flush()
            except Exception:
                pass

        channel = stdout.channel

        out_parts = []
        err_parts = []

        # Stream and collect output
        while True:
            if channel.recv_ready():
                data = stdout.read(channel.recv_ready()).decode("utf-8", errors="ignore")
                if data:
                    out_parts.append(data)
            if channel.recv_stderr_ready():
                edata = stderr.read(channel.recv_stderr_ready()).decode("utf-8", errors="ignore")
                if edata:
                    err_parts.append(edata)
            if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                break
            time.sleep(0.1)

        exit_code = channel.recv_exit_status()
        out = "".join(out_parts)
        err = "".join(err_parts)
        return out, err, exit_code
    finally:
        client.close()

def read_pzem_data(instrument):
    try:
        voltage = instrument.read_register(0x0000, number_of_decimals=2, functioncode=4)
        current = instrument.read_register(0x0001, number_of_decimals=2, functioncode=4)
        power_low = instrument.read_register(0x0002, functioncode=4)
        power_high = instrument.read_register(0x0003, functioncode=4)
        power = (power_high << 16) + power_low

        return voltage, current, power*0.1

    except minimalmodbus.ModbusException as e:
        print(f"Modbus error: {e}")
    except Exception as e:
        print(f"General error: {e}")

def measure_baseline(instrument):
    start_time=time.time()
    powers=[]
    while time.time()-start_time<=20:
        _, _, power = read_pzem_data(instrument)
        time.sleep(0.001)
        powers.append(power)
    avg_power=sum(powers)/len(powers)
    return avg_power
def measure_running(instrument, stop_event, output_list):
    while not stop_event.is_set():
        _, _, power = read_pzem_data(instrument)
        time.sleep(0.0001)
        output_list.append((time.time(), power))


def main():
    instrument = minimalmodbus.Instrument(PORT, DEVICE_ADDRESS)
    instrument.serial.baudrate = 9600
    instrument.serial.bytesize = 8
    instrument.serial.parity = serial.PARITY_NONE
    instrument.serial.stopbits = 2
    instrument.serial.timeout = 1


    # List of models to measure
    models = ["resnet18", "resnet50", "efficientnet_b1","efficientnet_b2","mobilenetv2_100"]
    batch_size=[1,2,4,8,16]
    mode_power=[0,1,2]

    combos = product(models, batch_size, mode_power)
    results={}

    for model, bs, mode in combos:
        cmd_run_model = f" nvpmodel -m {mode}"
        __, _, _ = cmd_ssh(host=IP, username=USERNAME, password=PASSWORD, command=cmd_run_model,use_sudo=True)
        print("Measuring baseline power consumption...")
        baseline_power=measure_baseline(instrument)
        print(f"Baseline power consumption: {baseline_power} W")
        print(f"\nMeasuring model: {model}")
        # Start measuring in a separate thread
        running_powers = []
        stop_event = threading.Event()
        t = threading.Thread(target=measure_running, args=(instrument, stop_event, running_powers), daemon=True)
        t.start()

        cmd_run_model = f"source venv/bin/activate && cd Energy_consumption && python3 run_model.py --model {model} --batch-size {bs}"
        out, err, exit_code = cmd_ssh(host=IP, username=USERNAME, password=PASSWORD, command=cmd_run_model)

        # Parse start and end times robustly
        m_start = re.search(r"start time:\s*([\d.]+)", out)
        m_end = re.search(r"end time:\s*([\d.]+)", out)
        if not m_start or not m_end:
            print(f"Could not parse start/end times for model {model}. stdout:\n{out}\nstderr:\n{err}")
            stop_event.set()
            t.join(timeout=2)
            results[model] = {"error": "parse_failed", "stdout": out, "stderr": err}
            continue

        start_time = float(m_start.group(1))
        end_time = float(m_end.group(1))

        # Stop the measuring thread and wait for it to finish
        stop_event.set()
        t.join(timeout=2)

        print("Start:", start_time)
        print("End:", end_time)

        samples = [s for s in running_powers if start_time < s[0] <=end_time]
        samples.sort(key=lambda x: x[0])

        print("Collected samples (within interval):", len(samples))

        # Integrate power (W) over time (s) using the trapezoidal rule to get energy in Joules
        energy_j = 0.0
        if len(samples) >= 2:
            for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
                dt = t1 - t0
                energy_j += 0.5 * (p0+ p1) * dt

            energy_j -= baseline_power * (end_time - start_time)
            energy_j /=500
            energy_wh = energy_j / 3600.0
            print(f"Model {model}: {energy_j:.6f} J ({energy_wh:.6f} Wh)")

            results[model] = {
                "energy_j": energy_j,
                "energy_wh": energy_wh,
                "samples": len(samples),
                "exit_code": exit_code,
                "batch_size": bs,
                "mode": mode,
                "latency": (end_time - start_time)/500
            }

            csv_file = "results.csv"
            file_exists = os.path.exists(csv_file)
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Model", "Energy (mJ)", "Latency (s)",  "error", "exit_code", "Batch_size", "Mode"])
                writer.writerow([
                    model,
                    (results.get(model, {}).get("energy_j") or 0) * 10**3,
                    results.get(model, {}).get("latency", 0),
                    results.get(model, {}).get("error", ""),
                    results.get(model, {}).get("exit_code", ""),
                    results.get(model, {}).get("batch_size", bs),
                    results.get(model, {}).get("mode", mode)
                ])
                f.flush()
                os.fsync(f.fileno())

        print("Sleep.....")
        time.sleep(10)

if __name__ == "__main__":
    main()
