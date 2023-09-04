#!/usr/bin/env python
"""
mlstemmer.py is a simple script that uses FastICA to remove the Drum component of a WAV file.

Wrote this to easily extract out drums from a sample or song that contains
other elements I want to use as sample chops.

Usage:

    python ./mlstemmer.py <wav_file>

This will output the components into seperate files. Didn't add detection to
determine which component (drums or whatever) corresponds to the output file.
Will do that after I experiment with the easiest way to get what I want.

Written with the help of ChatGPT. Wow I feel so trendy! Look at me I am doing AI stuffz! l33t lulz...
"""

import os
import sys
import pprint
from typing import List

import keras
import numpy as np

from sklearn.decomposition import FastICA
from scipy.io import wavfile


def main(args):

    if len(args) != 2:
        print(f"Usage: ./{args[0]} <input_wav_file>")
        sys.exit(1)

    # Open wavfile
    wav_file = args[1]
    print("Attempting to read WAV file.")
    if wav_file.endswith(".wav") == False:
        print("Input file must be a .wav file.")
        sys.exit(1)

    sample_rate, data = wavfile.read(wav_file)
    # Check if the audio is stereo
    if len(data.shape) != 2 or data.shape[1] < 2:
        print("The input audio file must be stereo for this code to work.")
        sys.exit(1)

    # Fast ICA . Isolate two components. Drums and everything else for now.
    num_components = 2  # Number of sources to separate
    ica = FastICA(n_components=num_components)
    transformed_data = ica.fit_transform(data)

    # Check the shape of the output
    print("Shape of transformed_data: ", transformed_data.shape)

    # Write each separated component to a new WAV file
    for i in range(num_components):
        component = transformed_data[:, i]
        print("Writing component:", i)
        # Normalize and convert to int16
        component = (component / np.abs(component).max() * 32767).astype(np.int16)
        wavfile.write(f'isolated_component_{i+1}.wav', sample_rate, component)


if __name__ == "__main__":
    main(sys.argv)
