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
hop_duration = 20      # Hop size in seconds

# Initialize a DataFrame to store groove-related metrics
groove_results = []

# Process each file
for file in files:
    filepath = os.path.join(directory, file)
    y, sr = librosa.load(filepath, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    # Process the file in chunks
    for start in range(0, int(duration), hop_duration):
        end = min(start + window_duration, duration)
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        y_segment = y[start_sample:end_sample]

        # Calculate onset strength
        onset_env = librosa.onset.onset_strength(y=y_segment, sr=sr)
        onset_variability = np.std(onset_env)  # Variability of onset strength

        # Calculate tempo stability
        bpm, beat_frames = librosa.beat.beat_track(y=y_segment, sr=sr)
        beat_intervals = np.diff(beat_frames) / sr
        tempo_stability = np.std(beat_intervals)  # Variability of beat intervals

        # Append results
        groove_results.append({
            "File": file,
            "Start_Time": start,
            "End_Time": end,
            "Onset_Variability": onset_variability,
            "Tempo_Stability": tempo_stability,
            "BPM": bpm
        })

# Convert results to a DataFrame
groove_df = pd.DataFrame(groove_results)

# Save the DataFrame to a CSV file
groove_df.to_csv("groove_analysis.csv", index=False)

print("Groove analysis saved to 'groove_analysis.csv'.")

