import os
import librosa
import numpy as np
import pandas as pd

# Directory containing MP3 files
directory = "samples"

# List of all files in the directory
files = [f for f in os.listdir(directory) if f.endswith(".mp3")]

# Initialize dictionaries to store MFCC features and BPM
mfcc_features = {}
bpm_values = {}

# Compute MFCC features and BPM for all files
for file in files:
    filepath = os.path.join(directory, file)
    y, sr = librosa.load(filepath, sr=None)
    
    # MFCC Features
    mfcc_features[file] = librosa.feature.mfcc(y, sr=sr).mean(axis=1)
    
    # BPM (Tempo)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    bpm = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    bpm_values[file] = bpm[0]  # Use the first (most likely) BPM value

# Initialize similarity matrices
n = len(files)
mfcc_similarity_matrix = np.zeros((n, n))
bpm_similarity_matrix = np.zeros((n, n))

# Compute similarity between all pairs of files
for i in range(n):
    for j in range(n):
        # MFCC Similarity
        if i == j:
            mfcc_similarity_matrix[i, j] = 1.0  # Perfect similarity with itself
        else:
            mfcc1 = mfcc_features[files[i]]
            mfcc2 = mfcc_features[files[j]]
            mfcc_similarity_matrix[i, j] = np.dot(mfcc1, mfcc2) / (
                np.linalg.norm(mfcc1) * np.linalg.norm(mfcc2)
            )
        
        # BPM Similarity (normalized difference)
        bpm1 = bpm_values[files[i]]
        bpm2 = bpm_values[files[j]]
        bpm_similarity_matrix[i, j] = 1 - abs(bpm1 - bpm2) / max(bpm1, bpm2)

# Convert the matrices to DataFrames
mfcc_similarity_df = pd.DataFrame(mfcc_similarity_matrix, index=files, columns=files)
bpm_similarity_df = pd.DataFrame(bpm_similarity_matrix, index=files, columns=files)

# Print the similarity matrices
print("MFCC Similarity Matrix:")
print(mfcc_similarity_df)

print("\nBPM Similarity Matrix:")
print(bpm_similarity_df)

# Optionally save the matrices to CSV files
mfcc_similarity_df.to_csv("mfcc_similarity_matrix.csv")
bpm_similarity_df.to_csv("bpm_similarity_matrix.csv")


