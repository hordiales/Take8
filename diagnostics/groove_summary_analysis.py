import os
import librosa
import numpy as np
import pandas as pd

# Directory containing MP3 files
directory = "samples"

# List of all files in the directory
files = [f for f in os.listdir(directory) if f.endswith(".mp3")]

# Define parameters for the analysis
window_duration = 20  # Window size in seconds
window_duration = 12  # Window size in seconds
hop_duration = 20      # Hop size in seconds
hop_duration = 12      # Hop size in seconds

# Initialize dictionaries for results
bpm_results = {}
onset_var_results = {}
tempo_stab_results = {}

# Process each file
for file in files:
    filepath = os.path.join(directory, file)
    y, sr = librosa.load(filepath, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    file_bpm = []
    file_onset_var = []
    file_tempo_stab = []

    # Process the file in chunks
    for start in range(0, int(duration), hop_duration):
        end = min(start + window_duration, duration)
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        y_segment = y[start_sample:end_sample]

        # Calculate BPM
        onset_env = librosa.onset.onset_strength(y=y_segment, sr=sr)
        bpm, beat_frames = librosa.beat.beat_track(y=y_segment, sr=sr)

        # Calculate onset variability
        onset_variability = np.std(onset_env)

        # Calculate tempo stability
        beat_intervals = np.diff(beat_frames) / sr
        tempo_stability = np.std(beat_intervals) if len(beat_intervals) > 0 else 0.0

        # Append BPM
        file_bpm.append(round(bpm, 1))

        # Classify onset variability
        if onset_variability < 0.2:
            file_onset_var.append("Low")
        elif onset_variability <= 0.5:
            file_onset_var.append("Moderate")
        else:
            file_onset_var.append("High")

        # Classify tempo stability
        if tempo_stability < 0.05:
            file_tempo_stab.append("Low")
        elif tempo_stability <= 0.15:
            file_tempo_stab.append("Moderate")
        else:
            file_tempo_stab.append("High")

    # Add results to dictionaries
    bpm_results[file] = file_bpm
    onset_var_results[file] = file_onset_var
    tempo_stab_results[file] = file_tempo_stab

# Convert results to DataFrames
time_ranges = [str(i) for i in range(0, len(max(bpm_results.values(), key=len)) * hop_duration, hop_duration)]
bpm_df = pd.DataFrame.from_dict(bpm_results, orient="index", columns=time_ranges)
onset_var_df = pd.DataFrame.from_dict(onset_var_results, orient="index", columns=time_ranges)
tempo_stab_df = pd.DataFrame.from_dict(tempo_stab_results, orient="index", columns=time_ranges)

# Save DataFrames to CSV files
bpm_df.to_csv("bpm_analysis.csv", index=True, index_label="File")
onset_var_df.to_csv("onset_variability_analysis.csv", index=True, index_label="File")
tempo_stab_df.to_csv("tempo_stability_analysis.csv", index=True, index_label="File")

print("Analysis completed. Outputs saved as 'bpm_analysis.csv', 'onset_variability_analysis.csv', and 'tempo_stability_analysis.csv'.")


