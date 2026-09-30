import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

class ECGSignalProcessor:
    """
    Biomedical engineering tool to filter raw ECG signals, 
    detect QRS complexes (R-peaks), and compute Heart Rate Variability metrics.
    """

    def __init__(self, raw_signal: np.ndarray, sampling_rate: float = 500.0):
        self.raw_signal = np.array(raw_signal, dtype=float)
        self.fs = sampling_rate
        self.filtered_signal = None
        self.r_peaks = None

    def bandpass_filter(self, lowcut: float = 0.5, highcut: float = 40.0, order: int = 2) -> np.ndarray:
        """Applies a Butterworth bandpass filter to remove baseline wander and high-frequency noise."""
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        self.filtered_signal = filtfilt(b, a, self.raw_signal)
        return self.filtered_signal

    def detect_r_peaks(self, min_distance_sec: float = 0.4) -> np.ndarray:
        """Detects R-peaks in the filtered signal based on height and distance thresholds."""
        if self.filtered_signal is None:
            self.bandpass_filter()
        
        distance = int(min_distance_sec * self.fs)
        # Dynamic height threshold set to 60% of the maximum peak amplitude
        height_threshold = np.max(self.filtered_signal) * 0.6
        
        peaks, _ = find_peaks(self.filtered_signal, distance=distance, height=height_threshold)
        self.r_peaks = peaks
        return self.r_peaks

    def calculate_hrv_metrics(self) -> dict:
        """Calculates Average Heart Rate (BPM) and RR-interval variability."""
        if self.r_peaks is None:
            self.detect_r_peaks()

        # Convert peak sample indices to time intervals in milliseconds
        peak_times = self.r_peaks / self.fs
        rr_intervals = np.diff(peak_times) * 1000.0  # in ms

        mean_rr = np.mean(rr_intervals)
        heart_rate = 60000.0 / mean_rr
        sdnn = np.std(rr_intervals)  # Standard deviation of NN intervals

        return {
            'heart_rate_bpm': heart_rate,
            'mean_rr_ms': mean_rr,
            'sdnn_ms': sdnn,
            'total_beats': len(self.r_peaks)
        }

    def plot_analysis(self, output_filename: str = 'ecg_analysis.png'):
        """Plots the raw vs filtered ECG signal along with detected R-peaks."""
        if self.filtered_signal is None or self.r_peaks is None:
            self.detect_r_peaks()

        time = np.arange(len(self.raw_signal)) / self.fs

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # Plot 1: Raw Signal
        axes[0].plot(time, self.raw_signal, color='#e41a1c', alpha=0.7, linewidth=1)
        axes[0].set_title('Raw ECG Signal (With Noise & Baseline Wander)', fontweight='bold')
        axes[0].set_ylabel('Amplitude (mV)')
        axes[0].grid(True, linestyle=':', alpha=0.6)

        # Plot 2: Filtered Signal + R-Peaks
        axes[1].plot(time, self.filtered_signal, color='#2b5c8f', linewidth=1.2, label='Filtered Signal')
        axes[1].scatter(time[self.r_peaks], self.filtered_signal[self.r_peaks], 
                         color='#4daf4a', s=60, label='Detected R-Peaks', zorder=5)
        axes[1].set_title('Filtered ECG Signal with Detected R-Peaks', fontweight='bold')
        axes[1].set_xlabel('Time (Seconds)')
        axes[1].set_ylabel('Amplitude (mV)')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.savefig(output_filename, dpi=300)
        plt.close()
        print(f"[+] ECG analysis plot saved to {output_filename}")


if __name__ == '__main__':
    # Generate Synthetic ECG signal for testing (10 seconds at 500 Hz)
    fs = 500.0
    duration = 10.0
    t = np.linspace(0, duration, int(fs * duration))
    
    # Synthetic clean heart rhythm (~75 BPM = 1.25 Hz)
    clean_ecg = np.sin(2 * np.pi * 1.25 * t)**10  
    # Add baseline wander (low frequency noise) and high-frequency thermal noise
    baseline_noise = 0.5 * np.sin(2 * np.pi * 0.1 * t)
    random_noise = 0.1 * np.random.normal(size=len(t))
    synthetic_raw_ecg = clean_ecg + baseline_noise + random_noise

    print("=== ECG SIGNAL PROCESSOR EXECUTING ===")
    processor = ECGSignalProcessor(synthetic_raw_ecg, sampling_rate=fs)

    # Process and extract metrics
    processor.bandpass_filter(lowcut=0.5, highcut=40.0)
    processor.detect_r_peaks()
    metrics = processor.calculate_hrv_metrics()

    print("\n[Biomedical Metrics Derived]")
    print(f"  Calculated Heart Rate : {metrics['heart_rate_bpm']:.1f} BPM")
    print(f"  Mean RR Interval      : {metrics['mean_rr_ms']:.2f} ms")
    print(f"  SDNN (HRV metric)     : {metrics['sdnn_ms']:.2f} ms")
    print(f"  Total Peaks Detected  : {metrics['total_beats']}")

    # Save visual plot
    processor.plot_analysis()
    print("\n[✓] Signal processing pipeline complete.")
