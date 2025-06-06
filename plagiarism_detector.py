#!/usr/bin/env python3
"""
Audio Plagiarism Detection System

This comprehensive tool analyzes two audio files to detect potential plagiarism
by examining multiple musical aspects including melody, harmony, rhythm, and structure.

Features:
- Melodic similarity analysis
- Harmonic progression comparison
- Rhythmic pattern matching
- Structural similarity detection
- Tempo and key analysis
- Overall plagiarism risk assessment
"""

import os
import warnings

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class PlagiarismDetector:
    def __init__(self, hop_length=512, frame_length=2048, n_mfcc=13):
        """
        Initialize the plagiarism detector
        
        Args:
            hop_length: Number of samples between successive frames
            frame_length: Length of each frame for analysis
            n_mfcc: Number of MFCC coefficients to extract
        """
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.n_mfcc = n_mfcc
        
        # Plagiarism thresholds
        self.thresholds = {
            'high_risk': 0.85,      # Very likely plagiarism
            'medium_risk': 0.70,    # Suspicious similarity
            'low_risk': 0.55,       # Some similarity
            'minimal_risk': 0.40    # Little similarity
        }
        
        # Musical note mapping
        self.note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def extract_comprehensive_features(self, audio_file):
        """
        Extract comprehensive musical features from audio file
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Dictionary containing all extracted features
        """
        print(f"🎵 Extracting features from: {os.path.basename(audio_file)}")
        
        try:
            # Load audio
            y, sr = librosa.load(audio_file, duration=120)  # Limit to 2 minutes for efficiency
            
            # 1. SPECTRAL FEATURES
            # MFCC (Mel-frequency cepstral coefficients) - Timbre
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length)
            
            # Chroma features - Harmonic content
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
            
            # Spectral contrast - Texture
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.hop_length)
            
            # Tonnetz - Harmonic network features
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr, hop_length=self.hop_length)
            
            # 2. RHYTHMIC FEATURES
            # Tempo and beat tracking
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
            
            # Onset detection
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=self.hop_length)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)
            
            # 3. MELODIC FEATURES
            # Pitch extraction
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=self.hop_length, fmin=80, fmax=2000)
            
            # Extract fundamental frequency
            pitch_sequence = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                pitch_sequence.append(pitch if pitch > 0 else 0)
            
            # 4. STRUCTURAL FEATURES
            # Zero crossing rate - Voice/instrument characteristics
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)
            
            # Spectral centroid - Brightness
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)
            
            # Root Mean Square Energy
            rms = librosa.feature.rms(y=y, hop_length=self.hop_length)
            
            print(f"✓ Extracted features from {len(y)/sr:.1f}s of audio")
            
            return {
                'audio': y,
                'sample_rate': sr,
                'mfcc': mfcc,
                'chroma': chroma,
                'spectral_contrast': spectral_contrast,
                'tonnetz': tonnetz,
                'tempo': float(tempo) if isinstance(tempo, np.ndarray) else tempo,
                'beats': beats,
                'onset_times': onset_times,
                'pitch_sequence': np.array(pitch_sequence),
                'zcr': zcr,
                'spectral_centroid': spectral_centroid,
                'spectral_rolloff': spectral_rolloff,
                'rms': rms,
                'duration': len(y) / sr
            }
            
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return None

    def analyze_melodic_similarity(self, features1, features2):
        """
        Analyze melodic similarity between two audio files
        
        Args:
            features1, features2: Feature dictionaries from both files
            
        Returns:
            Dictionary with melodic similarity metrics
        """
        print("🎼 Analyzing melodic similarity...")
        
        # Compare pitch sequences
        pitch1 = features1['pitch_sequence']
        pitch2 = features2['pitch_sequence']
        
        # Normalize pitch sequences to same length
        min_len = min(len(pitch1), len(pitch2))
        pitch1_norm = pitch1[:min_len]
        pitch2_norm = pitch2[:min_len]
        
        # Remove zero pitches for comparison
        valid_indices = (pitch1_norm > 0) & (pitch2_norm > 0)
        if np.sum(valid_indices) == 0:
            return {'pitch_correlation': 0.0, 'pitch_similarity': 0.0}
        
        valid_pitch1 = pitch1_norm[valid_indices]
        valid_pitch2 = pitch2_norm[valid_indices]
        
        # Calculate pitch correlation
        if len(valid_pitch1) > 1:
            pitch_correlation, _ = pearsonr(valid_pitch1, valid_pitch2)
            pitch_correlation = abs(pitch_correlation)  # Use absolute value
        else:
            pitch_correlation = 0.0
        
        # Calculate pitch similarity using cosine similarity
        if len(valid_pitch1) > 0:
            pitch_similarity = 1 - cosine(valid_pitch1, valid_pitch2)
        else:
            pitch_similarity = 0.0
        
        # Ensure non-negative values
        pitch_correlation = max(0.0, pitch_correlation)
        pitch_similarity = max(0.0, pitch_similarity)
        
        return {
            'pitch_correlation': pitch_correlation,
            'pitch_similarity': pitch_similarity
        }

    def analyze_harmonic_similarity(self, features1, features2):
        """
        Analyze harmonic similarity using chroma features
        
        Args:
            features1, features2: Feature dictionaries from both files
            
        Returns:
            Dictionary with harmonic similarity metrics
        """
        print("🎹 Analyzing harmonic similarity...")
        
        # Compare chroma features (harmonic content)
        chroma1 = np.mean(features1['chroma'], axis=1)
        chroma2 = np.mean(features2['chroma'], axis=1)
        
        # Chroma similarity
        chroma_similarity = 1 - cosine(chroma1, chroma2)
        
        # Compare tonnetz features (harmonic network)
        tonnetz1 = np.mean(features1['tonnetz'], axis=1)
        tonnetz2 = np.mean(features2['tonnetz'], axis=1)
        
        tonnetz_similarity = 1 - cosine(tonnetz1, tonnetz2)
        
        # Ensure non-negative values
        chroma_similarity = max(0.0, chroma_similarity)
        tonnetz_similarity = max(0.0, tonnetz_similarity)
        
        return {
            'chroma_similarity': chroma_similarity,
            'tonnetz_similarity': tonnetz_similarity,
            'harmonic_score': (chroma_similarity + tonnetz_similarity) / 2
        }

    def analyze_rhythmic_similarity(self, features1, features2):
        """
        Analyze rhythmic similarity including tempo and beat patterns
        
        Args:
            features1, features2: Feature dictionaries from both files
            
        Returns:
            Dictionary with rhythmic similarity metrics
        """
        print("🥁 Analyzing rhythmic similarity...")
        
        # Tempo similarity
        tempo1 = features1['tempo']
        tempo2 = features2['tempo']
        
        tempo_diff = abs(tempo1 - tempo2)
        max_tempo = max(tempo1, tempo2)
        tempo_similarity = 1 - (tempo_diff / max_tempo) if max_tempo > 0 else 0
        
        # Onset pattern similarity
        onset1 = features1['onset_times']
        onset2 = features2['onset_times']
        
        # Calculate onset intervals
        if len(onset1) > 1 and len(onset2) > 1:
            intervals1 = np.diff(onset1)
            intervals2 = np.diff(onset2)
            
            # Normalize to same length
            min_len = min(len(intervals1), len(intervals2))
            if min_len > 0:
                intervals1_norm = intervals1[:min_len]
                intervals2_norm = intervals2[:min_len]
                
                # Calculate correlation of onset intervals
                if len(intervals1_norm) > 1:
                    onset_correlation, _ = pearsonr(intervals1_norm, intervals2_norm)
                    onset_correlation = abs(onset_correlation)
                else:
                    onset_correlation = 0.0
            else:
                onset_correlation = 0.0
        else:
            onset_correlation = 0.0
        
        # Ensure non-negative values
        tempo_similarity = max(0.0, tempo_similarity)
        onset_correlation = max(0.0, onset_correlation)
        
        return {
            'tempo_similarity': tempo_similarity,
            'onset_correlation': onset_correlation,
            'tempo1': tempo1,
            'tempo2': tempo2,
            'rhythmic_score': (tempo_similarity + onset_correlation) / 2
        }

    def analyze_timbral_similarity(self, features1, features2):
        """
        Analyze timbral similarity using MFCC and spectral features
        
        Args:
            features1, features2: Feature dictionaries from both files
            
        Returns:
            Dictionary with timbral similarity metrics
        """
        print("🎨 Analyzing timbral similarity...")
        
        # MFCC similarity (timbre)
        mfcc1 = np.mean(features1['mfcc'], axis=1)
        mfcc2 = np.mean(features2['mfcc'], axis=1)
        
        mfcc_similarity = 1 - cosine(mfcc1, mfcc2)
        
        # Spectral contrast similarity
        contrast1 = np.mean(features1['spectral_contrast'], axis=1)
        contrast2 = np.mean(features2['spectral_contrast'], axis=1)
        
        contrast_similarity = 1 - cosine(contrast1, contrast2)
        
        # Spectral centroid similarity
        centroid1 = np.mean(features1['spectral_centroid'])
        centroid2 = np.mean(features2['spectral_centroid'])
        
        centroid_diff = abs(centroid1 - centroid2)
        max_centroid = max(centroid1, centroid2)
        centroid_similarity = 1 - (centroid_diff / max_centroid) if max_centroid > 0 else 0
        
        # Ensure non-negative values
        mfcc_similarity = max(0.0, mfcc_similarity)
        contrast_similarity = max(0.0, contrast_similarity)
        centroid_similarity = max(0.0, centroid_similarity)
        
        return {
            'mfcc_similarity': mfcc_similarity,
            'spectral_contrast_similarity': contrast_similarity,
            'spectral_centroid_similarity': centroid_similarity,
            'timbral_score': (mfcc_similarity + contrast_similarity + centroid_similarity) / 3
        }

    def calculate_plagiarism_risk(self, melodic_results, harmonic_results, rhythmic_results, timbral_results):
        """
        Calculate overall plagiarism risk based on all similarity metrics
        
        Args:
            melodic_results, harmonic_results, rhythmic_results, timbral_results: Analysis results
            
        Returns:
            Dictionary with plagiarism assessment
        """
        print("⚖️ Calculating plagiarism risk...")
        
        # Extract key similarity scores
        melodic_score = (melodic_results['pitch_correlation'] + melodic_results['pitch_similarity']) / 2
        harmonic_score = harmonic_results['harmonic_score']
        rhythmic_score = rhythmic_results['rhythmic_score']
        timbral_score = timbral_results['timbral_score']
        
        # Weighted overall similarity
        # Melody and harmony are more important for plagiarism detection
        weights = {
            'melodic': 0.35,
            'harmonic': 0.30,
            'rhythmic': 0.20,
            'timbral': 0.15
        }
        
        overall_similarity = (
            melodic_score * weights['melodic'] +
            harmonic_score * weights['harmonic'] +
            rhythmic_score * weights['rhythmic'] +
            timbral_score * weights['timbral']
        )
        
        # Determine risk level
        if overall_similarity >= self.thresholds['high_risk']:
            risk_level = "HIGH RISK"
            risk_description = "Very likely plagiarism - Multiple musical elements are highly similar"
        elif overall_similarity >= self.thresholds['medium_risk']:
            risk_level = "MEDIUM RISK"
            risk_description = "Suspicious similarity - Significant musical overlap detected"
        elif overall_similarity >= self.thresholds['low_risk']:
            risk_level = "LOW RISK"
            risk_description = "Some similarity - Common musical elements found"
        elif overall_similarity >= self.thresholds['minimal_risk']:
            risk_level = "MINIMAL RISK"
            risk_description = "Little similarity - Few common musical elements"
        else:
            risk_level = "NO RISK"
            risk_description = "Very different - No significant musical similarity"
        
        return {
            'overall_similarity': overall_similarity,
            'risk_level': risk_level,
            'risk_description': risk_description,
            'component_scores': {
                'melodic': melodic_score,
                'harmonic': harmonic_score,
                'rhythmic': rhythmic_score,
                'timbral': timbral_score
            }
        }

    def detect_plagiarism(self, file1, file2):
        """
        Main plagiarism detection function
        
        Args:
            file1, file2: Paths to audio files to compare
            
        Returns:
            Comprehensive plagiarism analysis results
        """
        print("=" * 80)
        print("🚨 AUDIO PLAGIARISM DETECTION ANALYSIS")
        print("=" * 80)
        print("Comparing:")
        print(f"  File 1: {os.path.basename(file1)}")
        print(f"  File 2: {os.path.basename(file2)}")
        print("=" * 80)
        
        # Extract features from both files
        features1 = self.extract_comprehensive_features(file1)
        features2 = self.extract_comprehensive_features(file2)
        
        if features1 is None or features2 is None:
            print("❌ Failed to extract features from one or both files")
            return None
        
        # Perform all similarity analyses
        melodic_results = self.analyze_melodic_similarity(features1, features2)
        harmonic_results = self.analyze_harmonic_similarity(features1, features2)
        rhythmic_results = self.analyze_rhythmic_similarity(features1, features2)
        timbral_results = self.analyze_timbral_similarity(features1, features2)
        
        # Calculate plagiarism risk
        plagiarism_assessment = self.calculate_plagiarism_risk(
            melodic_results, harmonic_results, rhythmic_results, timbral_results
        )
        
        # Display comprehensive results
        self.display_results(melodic_results, harmonic_results, rhythmic_results, 
                           timbral_results, plagiarism_assessment)
        
        return {
            'melodic': melodic_results,
            'harmonic': harmonic_results,
            'rhythmic': rhythmic_results,
            'timbral': timbral_results,
            'plagiarism_assessment': plagiarism_assessment,
            'features1': features1,
            'features2': features2
        }

    def display_results(self, melodic, harmonic, rhythmic, timbral, assessment):
        """
        Display comprehensive analysis results
        """
        print("\n" + "=" * 80)
        print("📊 DETAILED SIMILARITY ANALYSIS")
        print("=" * 80)
        
        print("\n🎼 MELODIC SIMILARITY:")
        print(f"   Pitch Correlation:      {melodic['pitch_correlation']:.3f}")
        print(f"   Pitch Similarity:       {melodic['pitch_similarity']:.3f}")
        
        print("\n🎹 HARMONIC SIMILARITY:")
        print(f"   Chroma Similarity:      {harmonic['chroma_similarity']:.3f}")
        print(f"   Tonnetz Similarity:     {harmonic['tonnetz_similarity']:.3f}")
        print(f"   Harmonic Score:         {harmonic['harmonic_score']:.3f}")
        
        print("\n🥁 RHYTHMIC SIMILARITY:")
        print(f"   Tempo Similarity:       {rhythmic['tempo_similarity']:.3f}")
        print(f"   Onset Correlation:      {rhythmic['onset_correlation']:.3f}")
        print(f"   Rhythmic Score:         {rhythmic['rhythmic_score']:.3f}")
        print(f"   Tempos: {rhythmic['tempo1']:.1f} BPM vs {rhythmic['tempo2']:.1f} BPM")
        
        print("\n🎨 TIMBRAL SIMILARITY:")
        print(f"   MFCC Similarity:        {timbral['mfcc_similarity']:.3f}")
        print(f"   Spectral Contrast:      {timbral['spectral_contrast_similarity']:.3f}")
        print(f"   Spectral Centroid:      {timbral['spectral_centroid_similarity']:.3f}")
        print(f"   Timbral Score:          {timbral['timbral_score']:.3f}")
        
        print("\n" + "=" * 80)
        print("🚨 PLAGIARISM RISK ASSESSMENT")
        print("=" * 80)
        print(f"Overall Similarity Score: {assessment['overall_similarity']:.3f}")
        print(f"Risk Level: {assessment['risk_level']}")
        print(f"Assessment: {assessment['risk_description']}")
        
        print("\n📈 Component Breakdown:")
        for component, score in assessment['component_scores'].items():
            print(f"   {component.capitalize():12}: {score:.3f} ({self.get_similarity_description(score)})")
        
        print("\n" + "=" * 80)
        if assessment['overall_similarity'] >= self.thresholds['medium_risk']:
            print("⚠️  WARNING: High similarity detected!")
            print("   Consider further investigation for potential copyright issues.")
        else:
            print("✅ Low similarity - No immediate plagiarism concerns.")
        print("=" * 80)

    def get_similarity_description(self, score):
        """Get descriptive text for similarity score"""
        if score >= 0.8:
            return "Very High"
        elif score >= 0.6:
            return "High"
        elif score >= 0.4:
            return "Medium"
        elif score >= 0.2:
            return "Low"
        else:
            return "Very Low"

    def create_visualization(self, results, file1, file2):
        """
        Create comprehensive visualization of plagiarism analysis
        
        Args:
            results: Analysis results from detect_plagiarism
            file1, file2: Original file paths for labeling
        """
        if not results:
            print("❌ No results to visualize")
            return
        
        print("📊 Creating plagiarism analysis visualization...")
        
        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Component Similarity Scores
        components = list(results['plagiarism_assessment']['component_scores'].keys())
        scores = list(results['plagiarism_assessment']['component_scores'].values())
        
        bars = ax1.barh(components, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_xlabel('Similarity Score')
        ax1.set_title('Similarity by Musical Component', fontweight='bold')
        ax1.set_xlim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center', fontweight='bold')
        
        # Plot 2: Risk Level Gauge
        overall_score = results['plagiarism_assessment']['overall_similarity']
        risk_level = results['plagiarism_assessment']['risk_level']
        
        # Create a gauge-like visualization
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        ax2.plot(theta, r, 'k-', linewidth=2)
        ax2.fill_between(theta, 0, r, alpha=0.3, color='lightgray')
        
        # Color zones
        zones = [
            (0, 0.4, 'green', 'Safe'),
            (0.4, 0.55, 'yellow', 'Caution'),
            (0.55, 0.7, 'orange', 'Warning'),
            (0.7, 1.0, 'red', 'High Risk')
        ]
        
        for start, end, color, label in zones:
            mask = (theta >= start * np.pi) & (theta <= end * np.pi)
            ax2.fill_between(theta[mask], 0, r[mask], alpha=0.7, color=color, label=label)
        
        # Add needle for current score
        needle_angle = overall_score * np.pi
        ax2.plot([needle_angle, needle_angle], [0, 1], 'k-', linewidth=4)
        ax2.plot(needle_angle, 1, 'ko', markersize=8)
        
        ax2.set_xlim(0, np.pi)
        ax2.set_ylim(0, 1.2)
        ax2.set_title(f'Plagiarism Risk: {risk_level}\nScore: {overall_score:.3f}', fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        ax2.set_xticklabels(['0.0', '0.25', '0.50', '0.75', '1.0'])
        ax2.set_yticks([])
        
        # Plot 3: Detailed Metrics Heatmap
        metrics_data = [
            ['Pitch Correlation', results['melodic']['pitch_correlation']],
            ['Pitch Similarity', results['melodic']['pitch_similarity']],
            ['Chroma Similarity', results['harmonic']['chroma_similarity']],
            ['Tonnetz Similarity', results['harmonic']['tonnetz_similarity']],
            ['Tempo Similarity', results['rhythmic']['tempo_similarity']],
            ['Onset Correlation', results['rhythmic']['onset_correlation']],
            ['MFCC Similarity', results['timbral']['mfcc_similarity']],
            ['Spectral Contrast', results['timbral']['spectral_contrast_similarity']]
        ]
        
        metrics_names = [item[0] for item in metrics_data]
        metrics_values = [item[1] for item in metrics_data]
        
        # Create heatmap data
        heatmap_data = np.array(metrics_values).reshape(-1, 1)
        
        im = ax3.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax3.set_yticks(range(len(metrics_names)))
        ax3.set_yticklabels(metrics_names)
        ax3.set_xticks([0])
        ax3.set_xticklabels(['Similarity'])
        ax3.set_title('Detailed Similarity Metrics', fontweight='bold')
        
        # Add text annotations
        for i, value in enumerate(metrics_values):
            ax3.text(0, i, f'{value:.3f}', ha='center', va='center', 
                    color='white' if value < 0.5 else 'black', fontweight='bold')
        
        # Plot 4: Summary Text
        ax4.axis('off')
        summary_text = f"""
PLAGIARISM ANALYSIS SUMMARY

Files Compared:
• {os.path.basename(file1)}
• {os.path.basename(file2)}

Overall Similarity: {overall_score:.3f}
Risk Level: {risk_level}

Key Findings:
• Melodic Score: {results['plagiarism_assessment']['component_scores']['melodic']:.3f}
• Harmonic Score: {results['plagiarism_assessment']['component_scores']['harmonic']:.3f}  
• Rhythmic Score: {results['plagiarism_assessment']['component_scores']['rhythmic']:.3f}
• Timbral Score: {results['plagiarism_assessment']['component_scores']['timbral']:.3f}

Assessment:
{results['plagiarism_assessment']['risk_description']}
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        plt.tight_layout()
        
        # Save visualization
        output_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "plagiarism_analysis.png"
        )
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved as: {output_file}")
        plt.show()


def main():
    """Main function to run plagiarism detection"""
    # Initialize detector
    detector = PlagiarismDetector()
    
    # Define audio files
    audio_dir = os.path.dirname(os.path.abspath(__file__))
    file1 = os.path.join(audio_dir, "JACK - J97 ｜ TRẠM DỪNG CHÂN ｜ Track No.3 [iK-Cji6J73Q].mp3")
    file2 = os.path.join(audio_dir, "梦散之地 - 颜人中 [hNiUGst5SX8].mp3")
    
    # Check if files exist
    if not os.path.exists(file1):
        print(f"❌ File not found: {os.path.basename(file1)}")
        return
    
    if not os.path.exists(file2):
        print(f"❌ File not found: {os.path.basename(file2)}")
        return
    
    # Perform plagiarism detection
    results = detector.detect_plagiarism(file1, file2)
    
    if results:
        # Create visualization
        detector.create_visualization(results, file1, file2)


if __name__ == "__main__":
    main()
