# music-plagiarism-detector-transformers

This repository contains the code and models for detecting potential music plagiarism using Transformer architectures. It includes three different models:
- **BERTOnlyClassifier:** based solely on song lyrics (text).
- **ViTOnlyClassifier:** based solely on spectrograms of the audio (visual).
- **BERTWithViT:** a multimodal model that combines lyrics and spectrograms.


The **main goal** is to classify whether a track is:
- An original composition
- An original cover
- A non-original cover (plagiarism)



Make sure you have the following before running the project:
- **20250204_bd_tfg_canciones.csv:** CSV file containing song metadata.
- **/Audios/:** folder containing the original .mp3 files.
- **/Espectogramas/:** folder where spectrogram images will be stored (generated automatically).



**Author:** Laura Henríquez Cazorla

**Date:** June 2025

> **Note:** *This project is intended for execution in Google Colab, as it relies on GPU acceleration and Google Drive integration.*


