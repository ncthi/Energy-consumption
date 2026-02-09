class JtopMonitor:
    
    def __init__(self, baseline_duration: int = 20, sample_interval: float = 0.01,filename="energy.csv",logs=True):
        self.baseline_duration = baseline_duration
        self.sample_interval = sample_interval
        self.jetson = None
        self.logs = logs

        self.baseline_power: List[float] = []
        self.measurement_power: List[float] = []
        self.baseline_avg: Optional[float] = None

        self.is_measuring = False
        self.measurement_thread = None
        self.start_time = None
        self.end_time = None
        self.filename = filename
        self.inference_name=None
        self.results=None
        
        print(f"Initialize Energy Monitor with baseline {baseline_duration}s and interval {sample_interval}s")

        self._measure_baseline()
    
    def _measure_baseline(self):

        print(f"Start measuring baseline in {self.baseline_duration} s")
        print("Please make sure no tasks are running!")
        
        try:
            with jtop(interval=0.005) as jetson:
                start_baseline = time.time()
                samples_count = 0
                
                while time.time() - start_baseline < self.baseline_duration:
                        stats = jetson.stats
                        power_total = stats.get("Power TOT", 0)
                        self.baseline_power.append(power_total)
                        samples_count += 1
                        
                        # Hiển thị tiến trình
                        elapsed = time.time() - start_baseline
                        remaining = self.baseline_duration - elapsed
                        print(f"\rMeasuring baseline: {elapsed:.1f}s/{self.baseline_duration}s - "
                              f"Power: {power_total:.2f}mW - remaining: {remaining:.1f}s", end="")
                        time.sleep(0.005)

                
                # Tính toán baseline trung bình
                if self.baseline_power:
                    self.baseline_avg = sum(self.baseline_power) / len(self.baseline_power)
                    print(f"\nComplete baseline measurement!")
                    print(f"Baseline sample number: {samples_count}")
                    print(f"Average baseline energy: {self.baseline_avg:.2f}mW")
                    print(f"Min: {min(self.baseline_power):.2f}mW, Max: {max(self.baseline_power):.2f}mW")
                else:
                    print("\nError: Unable to measure baseline!")
                    
        except Exception as e:
            print(f"\nError when measuring baseline: {e}")
    
    def _measurement_worker(self,shared_list):
        try:
            with jtop(interval=0.005) as jetson:
                while self.is_measuring:
                    stats = jetson.stats
                    power_total = stats.get("Power TOT", 0)
                    shared_list.append(power_total)
                    time.sleep(0.005)
                    
        except Exception as e:
            print(f"\nError during measurement: {e}")
    
    def start(self,inference_name):
        if self.baseline_avg is None:
            print("Error: No baseline data yet. Please re-initialize!")
            return False
        
        if self.is_measuring:
            print("Measuring in progress. Please stop() before starting() again!")
            return False
        
        if self.logs: print("\nStart measuring energy consumption...")
        self.inference_name=inference_name
        self.measurement_power=mp.Manager().list()
        self.results = None
        self.is_measuring = True
        self.start_time = time.time()
        self.measurement_thread = mp.Process(target=self._measurement_worker,args=(self.measurement_power,))
        self.measurement_thread.start()

        
        return True
    
    def stop(self) -> Dict:
        if not self.is_measuring:
            print("No measurement process is running!")
            return {}

        if self.logs: print("\nStop measuring energy...")
        self.is_measuring = False
        self.end_time = time.time()

        if self.measurement_thread:
            self.measurement_thread.terminate()
            self.measurement_thread.join()
        
        return self._analyze_results()
    
    def _analyze_results(self) -> Dict:
        if not self.measurement_power or self.baseline_avg is None:
            return {}

        duration = self.end_time - self.start_time if self.start_time and self.end_time else 0
        avg_power = sum(self.measurement_power) / len(self.measurement_power)
        min_power = min(self.measurement_power)
        max_power = max(self.measurement_power)

        power_increase = avg_power - self.baseline_avg
        power_increase_percent = (power_increase / self.baseline_avg) * 100 if self.baseline_avg > 0 else 0
        

        total_energy_baseline = self.baseline_avg * duration / 3600  # mWh
        total_energy_measurement = avg_power * duration / 3600  # mWh
        extra_energy = total_energy_measurement - total_energy_baseline
        
        self.results = {
            'duration': duration,
            'baseline_avg_power': self.baseline_avg,
            'measurement_avg_power': avg_power,
            'min_power': min_power,
            'max_power': max_power,
            'power_increase': power_increase,
            'power_increase_percent': power_increase_percent,
            'total_energy_baseline': total_energy_baseline,
            'total_energy_measurement': total_energy_measurement,
            'extra_energy_consumed': extra_energy,
            'samples_count': len(self.measurement_power)
        }

        if self.logs:
            print("\n" + "="*60)
            print("ENERGY MEASUREMENT RESULTS")
            print("="*60)
            print(f"Measurement time:           {duration:.2f} s")
            print(f"Number of samples:              {self.results['samples_count']}")
            print(f"Average Baseline:    {self.baseline_avg:.2f} mW")
            print(f"Average energy:  {avg_power:.2f} mW")
            print(f"Minimum energy:   {min_power:.2f} mW")
            print(f"Maximum energy:      {max_power:.2f} mW")
            print(f"Increase from baseline:   {power_increase:.2f} mW ({power_increase_percent:.1f}%)")
            print(f"Total baseline energy: {total_energy_baseline:.4f} mWh")
            print(f"Total energy measured:     {total_energy_measurement:.4f} mWh")
            print(f"Additional energy consumption: {extra_energy:.4f} mWh")
            print("="*60)
        
        return self.results
    
    def get_baseline_info(self) -> Dict:
        if not self.baseline_power:
            return {}
        
        return {
            'baseline_avg': self.baseline_avg,
            'baseline_min': min(self.baseline_power),
            'baseline_max': max(self.baseline_power),
            'baseline_samples': len(self.baseline_power),
            'baseline_duration': self.baseline_duration
        }
    
    def save_results(self):
        try:
            import pandas as pd
            import os

            data_to_write = [
                {
                    'Timestampe': time.time(),
                    'Inference name': self.inference_name,
                    'Duration': self.results['duration'],
                    'Energy consumption (mWh)': self.results['extra_energy_consumed'],
                },
            ]
            df = pd.DataFrame(data_to_write)
            df.to_csv(self.filename,
                      index=False,
                      mode='a',
                      header=not os.path.isfile(self.filename),
                      encoding='utf-8-sig')
            
            print(f"Results saved to file: {self.filename}")
            
        except Exception as e:
            print(self.inference_name)
            print(self.results)
            print(f"Error saving file: {e}")
    def set_filepath(self,filename):
        self.filename=filename