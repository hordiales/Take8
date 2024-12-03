import os
import librosa
import numpy as np
import pandas as pd

# Directory containing MP3 files
directory = "samples"

# List of all files in the directory
files = [f for f in os.listdir(directory) if f.endswith(".mp3")]

# Initialize an empty dictionary to store MFCC features
mfcc_features = {}

# Compute MFCC features for all files
for file in files:
    filepath = os.path.join(directory, file)
    y, sr = librosa.load(filepath, sr=None)
    mfcc_features[file] = librosa.feature.mfcc(y, sr=sr).mean(axis=1)

# Initialize a similarity matrix
n = len(files)
similarity_matrix = np.zeros((n, n))

# Compute similarity between all pairs of files
for i in range(n):
    for j in range(n):
        if i == j:
            similarity_matrix[i, j] = 1.0  # Perfect similarity with itself
        else:
            mfcc1 = mfcc_features[files[i]]
            mfcc2 = mfcc_features[files[j]]
            similarity_matrix[i, j] = np.dot(mfcc1, mfcc2) / (
                np.linalg.norm(mfcc1) * np.linalg.norm(mfcc2)
            )

# Convert the matrix to a DataFrame for better readability
similarity_df = pd.DataFrame(similarity_matrix, index=files, columns=files)

# Print the similarity matrix
print(similarity_df)

# Optionally save the similarity matrix to a CSV file
similarity_df.to_csv("similarity_matrix.csv")

