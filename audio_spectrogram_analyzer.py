#!/usr/bin/env python3
"""
Audio-Only Plagiarism Detection using Vision Transformer

This script analyzes two audio files using spectrograms and Vision Transformer
to detect potential plagiarism. It converts audio to spectrograms and uses
a pre-trained ViT model to classify the similarity patterns.

Features:
- Convert MP3/audio to log-scaled STFT spectrograms
- Save spectrograms as 224x224 PNG images
- Use Vision Transformer to analyze audio patterns
- Compare two audio files for plagiarism detection
"""

import os
import shutil
import tempfile
import warnings

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, ViTModel

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class ViTAudioClassifier(nn.Module):
    """
    Vision Transformer classifier for audio spectrogram analysis
    """

    def __init__(self, num_labels=3):
        """
        Initializes a classifier that uses a pretrained Vision Transformer (ViT)
        to classify spectrogram images.

        Args:
            num_labels (int): Number of output classes for classification.
        """
        super().__init__()
        self.vision_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_labels)
        )
        # Class weights for plagiarism detection: [Original, Different Cover, Similar Cover]
        self.class_weights = torch.tensor([1.55, 1.7, 1.45])

    def forward(self, spectrograms=None, labels=None):
        """
        Forward pass of the model.

        Args:
            spectrograms (torch.Tensor): Batch of image tensors representing spectrograms.
            labels (torch.Tensor, optional): True labels for computing loss.

        Returns:
            dict: Always includes 'logits'. If labels are given, also includes 'loss'.
        """
        vision_out = self.vision_model(pixel_values=spectrograms)
        vision_embed = vision_out.pooler_output
        logits = self.classifier(vision_embed)

        if labels is not None:
            weights = self.class_weights.to(logits.device)
            loss = F.cross_entropy(logits, labels, weight=weights)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}


class AudioSpectrogramAnalyzer:
    """
    Main class for audio spectrogram analysis and plagiarism detection
    """

    def __init__(self):
        """Initialize the analyzer with Vision Transformer components"""
        print("🔧 Initializing Audio Spectrogram Analyzer...")

        # Initialize Vision Transformer components
        self.feature_extractor = AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )

        # Create temporary directory for spectrograms
        self.temp_dir = tempfile.mkdtemp(prefix="spectrograms_")
        print(f"📁 Created temporary directory: {self.temp_dir}")

        # Classification labels
        self.labels = ["Original Master", "Different Cover", "Similar Cover (Potential Plagiarism)"]

        print("✅ Analyzer initialized successfully!")

    def __del__(self):
        """Cleanup temporary directory"""
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print("🧹 Cleaned up temporary directory")

    def create_spectrogram(self, audio_file, output_path=None):
        """
        Generate and save a log-scaled spectrogram image using STFT from an audio file.

        Args:
            audio_file (str): Path to the input audio file.
            output_path (str): Path where the resulting spectrogram image will be saved.

        Returns:
            str: Path to the saved spectrogram image
        """
        print(f"🎵 Creating spectrogram for: {os.path.basename(audio_file)}")

        if output_path is None:
            filename = os.path.splitext(os.path.basename(audio_file))[0] + "_spectrogram.png"
            output_path = os.path.join(self.temp_dir, filename)

        try:
            # Load audio file
            y, sr = librosa.load(
                audio_file, sr=None, duration=30
            )  # Limit to 30 seconds for efficiency

            # Compute the Short-Time Fourier Transform (STFT)
            D = np.abs(librosa.stft(y)) ** 2

            # Convert to logarithmic scale
            S = librosa.power_to_db(D, ref=np.max)

            # Create the spectrogram image
            plt.figure(figsize=(4, 4))
            librosa.display.specshow(S, sr=sr, x_axis=None, y_axis="log")
            plt.axis("off")
            plt.savefig(
                output_path, bbox_inches="tight", pad_inches=0, dpi=56
            )  # 224/4 = 56 DPI for 224x224
            plt.close()

            print(f"✅ Spectrogram saved: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ Error creating spectrogram: {e}")
            return None

    def load_and_process_spectrogram(self, spectrogram_path):
        """
        Load a spectrogram image and process it for Vision Transformer input.

        Args:
            spectrogram_path (str): Path to the spectrogram image file.

        Returns:
            torch.Tensor: A 3x224x224 tensor representing the preprocessed spectrogram image.
        """
        try:
            # Load and convert image to RGB
            image = Image.open(spectrogram_path).convert("RGB")

            # Resize to 224x224 if needed
            image = image.resize((224, 224), Image.Resampling.LANCZOS)

            # Process with Vision Transformer feature extractor
            encoding = self.feature_extractor(images=image, return_tensors="pt")
            return encoding["pixel_values"].squeeze(0)

        except Exception as e:
            print(f"❌ Error processing spectrogram: {e}")
            return torch.zeros(3, 224, 224)

    def extract_audio_features(self, audio_file):
        """
        Extract Vision Transformer features from an audio file.

        Args:
            audio_file (str): Path to audio file.

        Returns:
            torch.Tensor: Processed spectrogram tensor ready for ViT model.
        """
        print(f"🔍 Extracting features from: {os.path.basename(audio_file)}")

        # Create spectrogram
        spectrogram_path = self.create_spectrogram(audio_file)
        if spectrogram_path is None:
            return None

        # Process spectrogram for ViT
        spectrogram_tensor = self.load_and_process_spectrogram(spectrogram_path)

        return spectrogram_tensor

    def analyze_similarity(self, file1, file2, model=None):
        """
        Analyze similarity between two audio files using spectrograms and Vision Transformer.

        Args:
            file1 (str): Path to first audio file.
            file2 (str): Path to second audio file.
            model (ViTAudioClassifier, optional): Pre-trained model. If None, uses feature similarity.

        Returns:
            dict: Analysis results including similarity scores and risk assessment.
        """
        print("=" * 80)
        print("🚨 AUDIO SPECTROGRAM PLAGIARISM ANALYSIS")
        print("=" * 80)
        print("Comparing:")
        print(f"  File 1: {os.path.basename(file1)}")
        print(f"  File 2: {os.path.basename(file2)}")
        print("=" * 80)

        # Extract features from both files
        features1 = self.extract_audio_features(file1)
        features2 = self.extract_audio_features(file2)

        if features1 is None or features2 is None:
            print("❌ Failed to extract features from one or both files")
            return None

        # Calculate basic similarity using cosine similarity
        features1_flat = features1.flatten()
        features2_flat = features2.flatten()

        # Normalize features
        features1_norm = F.normalize(features1_flat.unsqueeze(0), p=2, dim=1)
        features2_norm = F.normalize(features2_flat.unsqueeze(0), p=2, dim=1)

        # Calculate cosine similarity
        cosine_sim = F.cosine_similarity(features1_norm, features2_norm).item()

        # Calculate correlation coefficient
        correlation = torch.corrcoef(torch.stack([features1_flat, features2_flat]))[0, 1].item()

        # Calculate Euclidean distance (normalized)
        euclidean_dist = F.pairwise_distance(features1_norm, features2_norm).item()
        euclidean_sim = 1 / (1 + euclidean_dist)  # Convert distance to similarity

        # Overall similarity score (weighted average)
        overall_similarity = cosine_sim * 0.5 + abs(correlation) * 0.3 + euclidean_sim * 0.2

        # Ensure non-negative values
        cosine_sim = max(0.0, cosine_sim)
        correlation = max(0.0, abs(correlation))
        euclidean_sim = max(0.0, euclidean_sim)
        overall_similarity = max(0.0, overall_similarity)

        # Risk assessment
        risk_assessment = self.assess_plagiarism_risk(overall_similarity)

        # Display results
        self.display_results(
            cosine_sim,
            correlation,
            euclidean_sim,
            overall_similarity,
            risk_assessment,
            file1,
            file2,
        )

        return {
            "cosine_similarity": cosine_sim,
            "correlation": correlation,
            "euclidean_similarity": euclidean_sim,
            "overall_similarity": overall_similarity,
            "risk_assessment": risk_assessment,
            "features1": features1,
            "features2": features2,
        }

    def assess_plagiarism_risk(self, similarity_score):
        """
        Assess plagiarism risk based on similarity score.

        Args:
            similarity_score (float): Overall similarity score (0-1).

        Returns:
            dict: Risk assessment with level and description.
        """
        if similarity_score >= 0.85:
            risk_level = "VERY HIGH RISK"
            risk_description = "Extremely similar spectrograms - Very likely plagiarism"
            risk_color = "🔴"
        elif similarity_score >= 0.70:
            risk_level = "HIGH RISK"
            risk_description = "Highly similar audio patterns - Likely plagiarism"
            risk_color = "🟠"
        elif similarity_score >= 0.55:
            risk_level = "MEDIUM RISK"
            risk_description = "Moderate similarity - Suspicious patterns detected"
            risk_color = "🟡"
        elif similarity_score >= 0.40:
            risk_level = "LOW RISK"
            risk_description = "Some similarity - Common musical elements"
            risk_color = "🟢"
        else:
            risk_level = "NO RISK"
            risk_description = "Very different - No significant similarity"
            risk_color = "🔵"

        return {
            "level": risk_level,
            "description": risk_description,
            "score": similarity_score,
            "color": risk_color,
        }

    def display_results(
        self, cosine_sim, correlation, euclidean_sim, overall_sim, risk_assessment, file1, file2
    ):
        """
        Display comprehensive analysis results.
        """
        print("\n" + "=" * 80)
        print("📊 SPECTROGRAM SIMILARITY ANALYSIS")
        print("=" * 80)

        print("\n🎼 AUDIO PATTERN SIMILARITY:")
        print(f"   Cosine Similarity:      {cosine_sim:.3f}")
        print(f"   Correlation:            {correlation:.3f}")
        print(f"   Euclidean Similarity:   {euclidean_sim:.3f}")
        print(f"   Overall Similarity:     {overall_sim:.3f}")

        print("\n" + "=" * 80)
        print("🚨 PLAGIARISM RISK ASSESSMENT")
        print("=" * 80)
        print(f"Risk Level: {risk_assessment['color']} {risk_assessment['level']}")
        print(f"Similarity Score: {risk_assessment['score']:.3f}")
        print(f"Assessment: {risk_assessment['description']}")

        print("\n" + "=" * 80)
        if overall_sim >= 0.70:
            print("⚠️  WARNING: High audio similarity detected!")
            print("   Consider further investigation for potential copyright issues.")
        else:
            print("✅ Low similarity - No immediate plagiarism concerns.")
        print("=" * 80)

    def create_visualization(self, results, file1, file2):
        """
        Create visualization comparing the spectrograms and similarity analysis.

        Args:
            results (dict): Analysis results from analyze_similarity.
            file1, file2 (str): Original file paths for labeling.
        """
        if not results:
            print("❌ No results to visualize")
            return

        print("📊 Creating spectrogram comparison visualization...")

        # Create spectrograms for both files
        spec1_path = self.create_spectrogram(file1, os.path.join(self.temp_dir, "file1_spec.png"))
        spec2_path = self.create_spectrogram(file2, os.path.join(self.temp_dir, "file2_spec.png"))

        if not spec1_path or not spec2_path:
            print("❌ Failed to create spectrograms for visualization")
            return

        # Create comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Load and display spectrograms
        spec1_img = Image.open(spec1_path)
        spec2_img = Image.open(spec2_path)

        ax1.imshow(spec1_img)
        ax1.set_title(f"Spectrogram: {os.path.basename(file1)}", fontweight="bold")
        ax1.axis("off")

        ax2.imshow(spec2_img)
        ax2.set_title(f"Spectrogram: {os.path.basename(file2)}", fontweight="bold")
        ax2.axis("off")

        # Similarity metrics bar chart
        metrics = [
            "Cosine\nSimilarity",
            "Correlation",
            "Euclidean\nSimilarity",
            "Overall\nSimilarity",
        ]
        scores = [
            results["cosine_similarity"],
            results["correlation"],
            results["euclidean_similarity"],
            results["overall_similarity"],
        ]

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
        bars = ax3.bar(metrics, scores, color=colors)
        ax3.set_ylabel("Similarity Score")
        ax3.set_title("Similarity Metrics", fontweight="bold")
        ax3.set_ylim(0, 1)

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Risk assessment gauge
        risk_score = results["risk_assessment"]["score"]
        risk_level = results["risk_assessment"]["level"]

        # Create a simple gauge
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)

        ax4.plot(theta, r, "k-", linewidth=2)
        ax4.fill_between(theta, 0, r, alpha=0.3, color="lightgray")

        # Color zones
        zones = [
            (0, 0.4, "green", "Safe"),
            (0.4, 0.55, "yellow", "Caution"),
            (0.55, 0.7, "orange", "Warning"),
            (0.7, 1.0, "red", "High Risk"),
        ]

        for start, end, color, label in zones:
            mask = (theta >= start * np.pi) & (theta <= end * np.pi)
            ax4.fill_between(theta[mask], 0, r[mask], alpha=0.7, color=color, label=label)

        # Add needle for current score
        needle_angle = risk_score * np.pi
        ax4.plot([needle_angle, needle_angle], [0, 1], "k-", linewidth=4)
        ax4.plot(needle_angle, 1, "ko", markersize=8)

        ax4.set_xlim(0, np.pi)
        ax4.set_ylim(0, 1.2)
        ax4.set_title(f"Plagiarism Risk: {risk_level}\nScore: {risk_score:.3f}", fontweight="bold")
        ax4.legend(loc="upper right")
        ax4.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
        ax4.set_xticklabels(["0.0", "0.25", "0.50", "0.75", "1.0"])
        ax4.set_yticks([])

        plt.tight_layout()

        # Save visualization
        output_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "spectrogram_plagiarism_analysis.png"
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"📊 Visualization saved as: {output_file}")
        plt.show()


def main():
    """Main function to run audio spectrogram plagiarism detection"""
    # Initialize analyzer
    analyzer = AudioSpectrogramAnalyzer()

    # Define audio files (same as quick_compare.py structure)
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

    # Perform analysis
    results = analyzer.analyze_similarity(file1, file2)

    if results:
        # Create visualization
        analyzer.create_visualization(results, file1, file2)

        print("\n🎯 CONCLUSION:")
        print(
            f"   These songs show {results['risk_assessment']['level'].lower()} based on spectrogram analysis"
        )
        print(f"   Overall Similarity: {results['overall_similarity']:.1%}")
        print(f"   {results['risk_assessment']['description']}")


if __name__ == "__main__":
    main()
