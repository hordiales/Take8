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

# Initialize a DataFrame to store BPM results
bpm_results = []

# Process each file
for file in files:
    filepath = os.path.join(directory, file)
    y, sr = librosa.load(filepath, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    # Process the file in 20-second chunks
    for start in range(0, int(duration), hop_duration):
        end = min(start + window_duration, duration)
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        y_segment = y[start_sample:end_sample]

        # Calculate BPM for the segment
        onset_env = librosa.onset.onset_strength(y=y_segment, sr=sr)
        bpm = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)

        # Append the result to the list
        bpm_results.append({
            "File": file,
            "Start_Time": start,
            "End_Time": end,
            "BPM": bpm[0]  # Use the first (most likely) BPM value
        })

# Convert results to a DataFrame
bpm_df = pd.DataFrame(bpm_results)

# Save the DataFrame to a CSV file
bpm_df.to_csv("bpm_analysis.csv", index=False)

print("BPM analysis saved to 'bpm_analysis.csv'.")


