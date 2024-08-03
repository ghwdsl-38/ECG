import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy import stats

# Step 1: Read record 100 from MIT-BIH database
record = wfdb.rdrecord('100', pn_dir='mitdb')
annotation = wfdb.rdann('100', 'atr', pn_dir='mitdb')

# Step 2: Adjust ECG signal for baseline and gain
ecg_signal = record.p_signal[:, 0]  # Assuming we are working with the first channel
baseline = record.baseline[0]  # Get baseline from record
gains = record.adc_gain[0]  # Get gains from record
adjusted_ecg_signal = (ecg_signal - baseline) / gains

# Step 3.5: Resample ECG signal to 150 Hz
original_fs = record.fs  # Sampling frequency of the original ECG signal
target_fs = 150  # Target sampling frequency

resampled_length = int(len(adjusted_ecg_signal) * target_fs / original_fs)
resampled_ecg_signal = resample(adjusted_ecg_signal, resampled_length)

# Step 3: Normalize adjusted ECG signal to [0, 1] using scipy
min_val = np.min(resampled_ecg_signal)
max_val = np.max(resampled_ecg_signal)
norm_ecg_signal = (resampled_ecg_signal - min_val) / (max_val - min_val)

# Step 4: Skip resampling and continue with R peaks extraction and segmentation

# Update R peaks to the original sampling rate
r_peaks = annotation.sample

# Step 5: Extract R peaks and their labels from annotations
labels = annotation.symbol

# Define the five classes to keep
valid_labels_set = {'N', 'A', 'V', 'F', 'U'}

# Filter R peaks and labels based on the valid labels set
valid_indices = [i for i, label in enumerate(labels) if label in valid_labels_set]
r_peaks = r_peaks[valid_indices]
labels = [labels[i] for i in valid_indices]

# Check if enough R peaks are detected
if len(r_peaks) < 2:
    print("Warning: Not enough valid R peaks detected. Skipping this record.")
    exit()

# Step 6: Signal fragment selection and zero-padding
fixed_length = 186  # Desired fixed length for each segment
segments = []
segment_labels = []

for i in range(len(r_peaks) - 1):  # Loop until the second last R peak
    current_r_peak = r_peaks[i]
    next_r_peak = r_peaks[i + 1]

    segment_start = int(current_r_peak * target_fs / original_fs)
    segment_end = int((current_r_peak + 1.2 * (next_r_peak - current_r_peak)) * target_fs / original_fs)

    # Ensure segment is within bounds
    if segment_end > len(norm_ecg_signal):
        segment_end = len(norm_ecg_signal)

    segment = norm_ecg_signal[segment_start:segment_end]

    # Zero-pad or trim to fixed length
    segment_length = len(segment)
    if segment_length < fixed_length:
        padded_segment = np.zeros(fixed_length)
        padded_segment[:segment_length] = segment
        segment = padded_segment
    elif segment_length > fixed_length:
        segment = segment[:fixed_length]

    segments.append(segment)
    segment_labels.append(labels[i])  # Use label corresponding to the current segment

# Convert lists to numpy arrays
segments = np.array(segments)
segment_labels = np.array(segment_labels)

# Print shapes for verification
print(f"Segments shape: {segments.shape}")
print(f"Labels shape: {segment_labels.shape}")

# Count the number of each label
unique_labels, counts = np.unique(segment_labels, return_counts=True)
label_counts = dict(zip(unique_labels, counts))
print(f"Label counts: {label_counts}")

# Optional: Plot the first segment and its corresponding label
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(norm_ecg_signal, label='ECG Signal')
plt.title('Normalized ECG Signal')
plt.xlabel('Sample number')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(np.arange(fixed_length), segments[0], label=f'Segment with label {segment_labels[0]}')
plt.title(f'Selected Segment with label {segment_labels[0]}')
plt.xlabel('Sample number')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()