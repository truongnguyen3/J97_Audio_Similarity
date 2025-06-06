#!/usr/bin/env python3
"""
Quick Audio Similarity Comparison

A simplified version for quick similarity analysis
"""

import librosa
import numpy as np
from scipy.spatial.distance import cosine
import os

def quick_audio_similarity(file1, file2):
    """Quick comparison using key audio features"""
    print(f"🎵 Comparing: {os.path.basename(file1)} vs {os.path.basename(file2)}")
    
    try:
        # Load audio files (first 30 seconds)
        y1, sr1 = librosa.load(file1, duration=30)
        y2, sr2 = librosa.load(file2, duration=30)
        
        # Extract MFCC features (most important for similarity)
        mfcc1 = np.mean(librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13), axis=1)
        mfcc2 = np.mean(librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13), axis=1)
        
        # Extract chroma features (harmonic content)
        chroma1 = np.mean(librosa.feature.chroma_stft(y=y1, sr=sr1), axis=1)
        chroma2 = np.mean(librosa.feature.chroma_stft(y=y2, sr=sr2), axis=1)
        
        # Extract tempo
        tempo1, _ = librosa.beat.beat_track(y=y1, sr=sr1)
        tempo2, _ = librosa.beat.beat_track(y=y2, sr=sr2)
        
        # Convert tempo to scalar if it's an array
        if isinstance(tempo1, np.ndarray):
            tempo1 = float(tempo1[0]) if len(tempo1) > 0 else 120.0
        if isinstance(tempo2, np.ndarray):
            tempo2 = float(tempo2[0]) if len(tempo2) > 0 else 120.0
        
        # Calculate similarities
        mfcc_similarity = 1 - cosine(mfcc1, mfcc2)
        chroma_similarity = 1 - cosine(chroma1, chroma2)
        tempo_similarity = 1 - abs(tempo1 - tempo2) / max(float(tempo1), float(tempo2))
        
        # Overall similarity (weighted average)
        overall_similarity = (mfcc_similarity * 0.5 + 
                            chroma_similarity * 0.3 + 
                            tempo_similarity * 0.2)
        
        # Ensure non-negative values
        mfcc_similarity = max(0, mfcc_similarity)
        chroma_similarity = max(0, chroma_similarity)
        tempo_similarity = max(0, tempo_similarity)
        overall_similarity = max(0, overall_similarity)
        
        # Display results
        print(f"\n📊 Similarity Results:")
        print(f"   Overall Similarity: {overall_similarity:.3f} ({get_similarity_level(overall_similarity)})")
        print(f"   Timbre Similarity:  {mfcc_similarity:.3f}")
        print(f"   Harmony Similarity: {chroma_similarity:.3f}")
        print(f"   Tempo Similarity:   {tempo_similarity:.3f}")
        print(f"\n🎼 Tempo Info:")
        print(f"   File 1 Tempo: {tempo1:.1f} BPM")
        print(f"   File 2 Tempo: {tempo2:.1f} BPM")
        
        return {
            'overall': overall_similarity,
            'timbre': mfcc_similarity,
            'harmony': chroma_similarity,
            'tempo': tempo_similarity
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def get_similarity_level(score):
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

if __name__ == "__main__":
    # Your audio files
    audio_dir = "/Users/truong.nguyen3/Documents/_work_samples/Audio_Similarity"
    file1 = os.path.join(audio_dir, "JACK - J97 ｜ TRẠM DỪNG CHÂN ｜ Track No.3 [iK-Cji6J73Q].mp3")
    file2 = os.path.join(audio_dir, "梦散之地 - 颜人中 [hNiUGst5SX8].mp3")
    
    # Run comparison
    similarity = quick_audio_similarity(file1, file2)
    
    if similarity:
        print(f"\n🎯 Conclusion: These songs are {get_similarity_level(similarity['overall']).lower()}")
        print(f"    Similarity Score: {similarity['overall']:.1%}")
