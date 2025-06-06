#!/usr/bin/env python3
"""
Audio Similarity Comparison Tool

This script compares two audio files using multiple similarity metrics:
- MFCC features (timbre similarity)
- Chroma features (harmonic similarity) 
- Spectral features (frequency content)
- Rhythm/tempo features
- Overall weighted similarity score
"""

import librosa
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import os
import sys

class AudioSimilarityAnalyzer:
    def __init__(self):
        self.features_cache = {}
    
    def extract_audio_features(self, audio_file):
        """Extract comprehensive audio features from an audio file"""
        print(f"Extracting features from: {os.path.basename(audio_file)}")
        
        try:
            # Load audio file
            y, sr = librosa.load(audio_file, duration=30)  # Load first 30 seconds for efficiency
            
            features = {}
            
            # MFCC features (Mel-frequency cepstral coefficients) - timbre/texture
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc'] = np.mean(mfcc, axis=1)
            features['mfcc_std'] = np.std(mfcc, axis=1)
            
            # Chroma features - harmonic/pitch content
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma'] = np.mean(chroma, axis=1)
            
            # Spectral features
            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # Rhythm features
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = tempo
            
            # RMS energy
            features['rms'] = np.mean(librosa.feature.rms(y=y))
            
            print(f"✓ Features extracted successfully")
            return features
            
        except Exception as e:
            print(f"✗ Error extracting features: {e}")
            return None
    
    def calculate_similarity(self, features1, features2):
        """Calculate similarity between two feature sets"""
        similarities = {}
        
        # MFCC similarity (most important for overall sound similarity)
        mfcc_sim = 1 - cosine(features1['mfcc'], features2['mfcc'])
        similarities['mfcc'] = max(0, mfcc_sim)  # Ensure non-negative
        
        # Chroma similarity (harmonic content)
        chroma_sim = 1 - cosine(features1['chroma'], features2['chroma'])
        similarities['chroma'] = max(0, chroma_sim)
        
        # Tempo similarity
        tempo_diff = abs(features1['tempo'] - features2['tempo'])
        max_tempo = max(features1['tempo'], features2['tempo'])
        tempo_sim = 1 - (tempo_diff / max_tempo) if max_tempo > 0 else 0
        similarities['tempo'] = tempo_sim
        
        # Spectral centroid similarity
        centroid_diff = abs(features1['spectral_centroid'] - features2['spectral_centroid'])
        max_centroid = max(features1['spectral_centroid'], features2['spectral_centroid'])
        centroid_sim = 1 - (centroid_diff / max_centroid) if max_centroid > 0 else 0
        similarities['spectral_centroid'] = centroid_sim
        
        # RMS energy similarity
        rms_diff = abs(features1['rms'] - features2['rms'])
        max_rms = max(features1['rms'], features2['rms'])
        rms_sim = 1 - (rms_diff / max_rms) if max_rms > 0 else 0
        similarities['rms'] = rms_sim
        
        # Overall weighted similarity
        # MFCC gets highest weight as it's most indicative of overall similarity
        weights = {
            'mfcc': 0.4,
            'chroma': 0.25,
            'tempo': 0.15,
            'spectral_centroid': 0.1,
            'rms': 0.1
        }
        
        overall_sim = sum(similarities[key] * weights[key] for key in weights.keys())
        similarities['overall'] = overall_sim
        
        return similarities
    
    def compare_audio_files(self, file1, file2):
        """Compare two audio files and return similarity metrics"""
        print("=" * 60)
        print("AUDIO SIMILARITY ANALYSIS")
        print("=" * 60)
        
        # Extract features from both files
        features1 = self.extract_audio_features(file1)
        features2 = self.extract_audio_features(file2)
        
        if features1 is None or features2 is None:
            print("✗ Failed to extract features from one or both files")
            return None
        
        # Calculate similarities
        similarities = self.calculate_similarity(features1, features2)
        
        # Display results
        print("\nSIMILARITY RESULTS:")
        print("-" * 40)
        print(f"Overall Similarity:      {similarities['overall']:.3f} ({self.get_similarity_description(similarities['overall'])})")
        print(f"Timbre Similarity (MFCC): {similarities['mfcc']:.3f}")
        print(f"Harmonic Similarity:     {similarities['chroma']:.3f}")
        print(f"Tempo Similarity:        {similarities['tempo']:.3f}")
        print(f"Spectral Similarity:     {similarities['spectral_centroid']:.3f}")
        print(f"Energy Similarity:       {similarities['rms']:.3f}")
        
        # Display additional info
        print("\nADDITIONAL INFO:")
        print("-" * 40)
        print(f"File 1 Tempo: {features1['tempo']:.1f} BPM")
        print(f"File 2 Tempo: {features2['tempo']:.1f} BPM")
        print(f"File 1 Spectral Centroid: {features1['spectral_centroid']:.0f} Hz")
        print(f"File 2 Spectral Centroid: {features2['spectral_centroid']:.0f} Hz")
        
        return similarities
    
    def get_similarity_description(self, score):
        """Convert similarity score to descriptive text"""
        if score >= 0.8:
            return "Very Similar"
        elif score >= 0.6:
            return "Similar"
        elif score >= 0.4:
            return "Somewhat Similar"
        elif score >= 0.2:
            return "Different"
        else:
            return "Very Different"
    
    def create_similarity_visualization(self, similarities, file1, file2):
        """Create a bar chart visualization of similarity metrics"""
        metrics = ['Overall', 'Timbre\n(MFCC)', 'Harmonic\n(Chroma)', 'Tempo', 'Spectral', 'Energy\n(RMS)']
        scores = [
            similarities['overall'],
            similarities['mfcc'],
            similarities['chroma'],
            similarities['tempo'],
            similarities['spectral_centroid'],
            similarities['rms']
        ]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(metrics, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.ylim(0, 1.1)
        plt.ylabel('Similarity Score', fontsize=12)
        plt.xlabel('Similarity Metrics', fontsize=12)
        plt.title(f'Audio Similarity Analysis\n{os.path.basename(file1)} vs {os.path.basename(file2)}', 
                 fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'similarity_analysis.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved as: {output_file}")
        plt.show()

def main():
    # Initialize analyzer
    analyzer = AudioSimilarityAnalyzer()
    
    # Define the audio files - use current directory
    audio_dir = os.path.dirname(os.path.abspath(__file__))
    file1 = os.path.join(audio_dir, "JACK - J97 ｜ TRẠM DỪNG CHÂN ｜ Track No.3 [iK-Cji6J73Q].mp3")
    file2 = os.path.join(audio_dir, "梦散之地 - 颜人中 [hNiUGst5SX8].mp3")
    
    # Check if files exist
    if not os.path.exists(file1):
        print(f"✗ File not found: {file1}")
        return
    
    if not os.path.exists(file2):
        print(f"✗ File not found: {file2}")
        return
    
    print(f"🎵 Comparing audio files:")
    print(f"   File 1: {os.path.basename(file1)}")
    print(f"   File 2: {os.path.basename(file2)}")
    
    # Perform comparison
    similarities = analyzer.compare_audio_files(file1, file2)
    
    if similarities:
        # Create visualization
        analyzer.create_similarity_visualization(similarities, file1, file2)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"The songs are: {analyzer.get_similarity_description(similarities['overall'])}")
        print(f"Overall similarity score: {similarities['overall']:.1%}")

if __name__ == "__main__":
    main()
