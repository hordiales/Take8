import os
import numpy as np
import pandas as pd
from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor
from essentia.standard import MonoLoader, RhythmExtractor2013, SpectralCentroidTime, Energy

# Directory containing MP3 files
directory = "samples"

# List of all files in the directory
files = [f for f in os.listdir(directory) if f.endswith(".mp3")]

# Parameters for analysis
window_duration = 20  # Window size in seconds
hop_duration = 20      # Hop size in seconds

# Initialize dictionaries to store results
syncopation_results = {}
spectral_centroid_results = {}
beat_strength_variance_results = {}
short_term_energy_results = {}

# Helper function to calculate beat strength variance
def calculate_beat_strength_variance(beat_strengths):
    return np.var(beat_strengths) if len(beat_strengths) > 1 else 0.0

# Process each file
for file in files:
    filepath = os.path.join(directory, file)
    print(f"Processing: {file}")

    # Load audio using Essentia
    audio = MonoLoader(filename=filepath)()
    sample_rate = 44100  # Default for Essentia's MonoLoader
    duration = len(audio) / sample_rate
    file_syncopation = []
    file_spectral_centroid = []
    file_beat_strength_variance = []
    file_short_term_energy = []

    # Process the file in chunks
    for start in range(0, int(duration), hop_duration):
        end = min(start + window_duration, duration)
        segment = audio[int(start * sample_rate):int(end * sample_rate)]

        # **1. Syncopation Analysis using Madmom**
        beat_processor = RNNBeatProcessor()(segment)
        beats = BeatTrackingProcessor(fps=100)(beat_processor)
        syncopation_score = len(beats) / (end - start)  # Syncopation density
        file_syncopation.append(round(syncopation_score, 4))

        # **2. Spectral Centroid (using Essentia)**
        spectral_centroid = SpectralCentroidTime()(segment)
        file_spectral_centroid.append(round(np.mean(spectral_centroid), 4))

        # **3. Beat Strength Variance**
        rhythm_extractor = RhythmExtractor2013(method="multifeature")
        bpm, confidence, _, beat_strengths, _ = rhythm_extractor(segment)
        beat_strength_variance = calculate_beat_strength_variance(beat_strengths)
        file_beat_strength_variance.append(round(beat_strength_variance, 4))

        # **4. Short-Term Energy**
        short_term_energy = Energy()(segment)
        file_short_term_energy.append(round(short_term_energy, 4))

    # Add results to dictionaries
    syncopation_results[file] = file_syncopation
    spectral_centroid_results[file] = file_spectral_centroid
    beat_strength_variance_results[file] = file_beat_strength_variance
    short_term_energy_results[file] = file_short_term_energy

# Convert results to DataFrames
time_ranges = [str(i) for i in range(0, len(max(syncopation_results.values(), key=len)) * hop_duration, hop_duration)]
syncopation_df = pd.DataFrame.from_dict(syncopation_results, orient="index", columns=time_ranges)
spectral_centroid_df = pd.DataFrame.from_dict(spectral_centroid_results, orient="index", columns=time_ranges)
beat_strength_variance_df = pd.DataFrame.from_dict(beat_strength_variance_results, orient="index", columns=time_ranges)
short_term_energy_df = pd.DataFrame.from_dict(short_term_energy_results, orient="index", columns=time_ranges)

# Save DataFrames to CSV files
syncopation_df.to_csv("syncopation_analysis.csv", index=True, index_label="File")
spectral_centroid_df.to_csv("spectral_centroid_analysis.csv", index=True, index_label="File")
beat_strength_variance_df.to_csv("beat_strength_variance_analysis.csv", index=True, index_label="File")
short_term_energy_df.to_csv("short_term_energy_analysis.csv", index=True, index_label="File")

print("Analysis completed. Outputs saved as:")
print("- 'syncopation_analysis.csv'")
print("- 'spectral_centroid_analysis.csv'")
print("- 'beat_strength_variance_analysis.csv'")
print("- 'short_term_energy_analysis.csv'")


