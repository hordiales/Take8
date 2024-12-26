# Take8
Live recording take analysis.

# Requirements 

    $ pip install -i requirements.txt

If you have problems with madmon, try with:

    $ pip install git+https://github.com/CPJKU/madmom


Note: Conda env is recommended.

# Use

Put your takes in .mp3 files on "samples" directory
Run the scripts
Output is in .csv files

# Analyis

CSV outputs:

* Syncopation Analysis: Provides syncopation density for each time range.

* Spectral Centroid Analysis: Average spectral centroid (brightness of the sound) for each time range.

* Beat Strength Variance: Variance of beat strengths for each time range.
* Short-Term Energy Analysis: Average short-term energy for each time range.


# Sample

8 tracks bpm analysis (window=20seconds)
![](sample-8tracks-bpm-analysis-20s.png)