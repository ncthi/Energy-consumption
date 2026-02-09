import sys
import time
import usb.core
import usb.util
import threading
import pandas as pd
import os
import timm
import torch
from tqdm import tqdm

#FNB48
VID = 0x0483
PID_FNB48 = 0x003A

PID_C1 = 0x003B

# FNB58
VID_FNB58 = 0x2E3C
PID_FNB58 = 0x5558

# FNB48S
VID_FNB48S = 0x2E3C
PID_FNB48S = 0x0049


class FNBTool:
    def __init__(self,time_interval=0.02,file_path="energy.csv"):
        self._time_interval = time_interval
        self._file_path = file_path
        self._find_device()
        assert self._dev, "Device not found"
        self._interface_hid_num = self._find_hid_interface_num()
        self._ensure_all_interfaces_not_busy()

        cfg = self._dev.get_active_configuration()
        intf = cfg[(self._interface_hid_num, 0)]
        self._ep_out = usb.util.find_descriptor(
            intf,
            custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT,
        )

        self._ep_in = usb.util.find_descriptor(
            intf,
            # match the first IN endpoint
            custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN,
        )

        assert self._ep_in
        assert self._ep_out
        self._running=False
        self._energy_init=0
        self._power=0
        self._energy=0
        self._n_measurements=0
        self._measure_thread=None
        self._inference_name=""
        self._start_time=0
        self._request_data()
        self._init()

    def _find_device(self):
        self._is_fnb58_or_fnb48s = False
        self._dev = usb.core.find(idVendor=VID, idProduct=PID_FNB48)
        if self._dev is None:
            self._dev = usb.core.find(idVendor=VID, idProduct=PID_C1)
        if self._dev is None:
            self._dev = usb.core.find(idVendor=VID_FNB58, idProduct=PID_FNB58)
            if self._dev:
                self._is_fnb58_or_fnb48s = True
        if self._dev is None:
            self._dev = usb.core.find(idVendor=VID_FNB48S, idProduct=PID_FNB48S)
            if self._dev:
                self._is_fnb58_or_fnb48s = True
    def _find_hid_interface_num(self):
        for cfg in self._dev:
            for interface in cfg:
                if interface.bInterfaceClass == 0x03:
                    return interface.bInterfaceNumber
    def _ensure_all_interfaces_not_busy(self):
        for cfg in self._dev:
            for interface in cfg:
                if self._dev.is_kernel_driver_active(interface.bInterfaceNumber):
                    try:
                        self._dev.detach_kernel_driver(interface.bInterfaceNumber)
                    except usb.core.USBError as e:
                        print(f"Could not detatch kernel driver from interface({interface.bInterfaceNumber}): {e}", file=sys.stderr)
                        sys.exit(1)
    def _request_data(self):
        self._ep_out.write(b"\xaa\x81" + b"\x00" * 61 + b"\x8e")
        time.sleep(0.1)
        self._ep_out.write(b"\xaa\x82" + b"\x00" * 61 + b"\x96")
        time.sleep(0.1)
        if not self._is_fnb58_or_fnb48s:
            self._ep_out.write(b"\xaa\x83" + b"\x00" * 61 + b"\x9e")
    def _decode(self, data):
        packet_type = data[1]
        if packet_type != 0x04:
            return None

        for i in range(4):
            offset = 2 + 15 * i
            voltage = (
                              data[offset + 3] * 256 * 256 * 256
                              + data[offset + 2] * 256 * 256
                              + data[offset + 1] * 256
                              + data[offset + 0]
                      ) / 100000
            current = (
                              data[offset + 7] * 256 * 256 * 256
                              + data[offset + 6] * 256 * 256
                              + data[offset + 5] * 256
                              + data[offset + 4]
                      ) / 100000
            return  voltage * current

    def start(self,name):
        self._inference_name=name
        self._request_data()
        assert not self._running, "Measurement process is running, you have to stop before start"
        self._running = True
        self._power=0
        self._energy=0
        self._n_measurements=0
        self._measure_thread = threading.Thread(target=self._read_data)
        self._measure_thread.start()
        self._start_time=time.time()


    def stop(self):
        assert self._running, "Measurement process is not running"
        self._running = False
        self._measure_thread=None
        data = self._ep_in.read(size_or_buffer=64, timeout=1000)
        power = self._decode(data)
        if power is not None:
            self._power+=power
            self._n_measurements+=1
        if self._n_measurements != 0:  self._power=self._power/self._n_measurements-self._power_init
        duration=time.time()-self._start_time
        self._energy= self._power*duration/3.6
        data_to_write = [
            {
                'Start time': self._start_time,
                'Inference name': self._inference_name,
                'Duration': duration,
                'Energy consumption (mWh)': self._energy
            },
        ]
        df = pd.DataFrame(data_to_write)
        df.to_csv(self._file_path,
                  index=False,
                  mode='a',
                  header=not os.path.isfile(self._file_path),
                  encoding='utf-8-sig')
    def _read_data(self):
        continue_time = time.time()
        while self._running:
            if time.time() >= continue_time:
                continue_time = time.time() + 1
                self._ep_out.write(b"\xaa\x83" + b"\x00" * 61 + b"\x9e")
            data = self._ep_in.read(size_or_buffer=64, timeout=1000)
            power = self._decode(data)
            if power is not None:
                self._power+=power
                self._n_measurements+=1
            time.sleep(self._time_interval)

    def _init(self):
        self._running = True
        self._measure_thread = threading.Thread(target=self._read_data)
        self._measure_thread.start()
        time.sleep(20)
        self._running = False
        self._power_init=self._power/self._n_measurements

    def set_filepath(self,file_path):
        self._file_path = file_path