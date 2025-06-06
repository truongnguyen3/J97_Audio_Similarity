#!/usr/bin/env python3
"""
Continuous Same Note Detection Tool

This script analyzes two audio files to detect continuous segments where
they play the same musical notes. It uses pitch detection and note mapping
to identify matching melodic patterns.

Features:
- Pitch extraction using librosa
- Note conversion (Hz to musical notes)
- Continuous same note detection
- Temporal alignment analysis
- Visual representation of matching segments
"""

import os

import librosa
import matplotlib.pyplot as plt
import numpy as np


class SameNoteDetector:
    def __init__(self, hop_length=512, frame_length=2048):
        """
        Initialize the same note detector

        Args:
            hop_length: Number of samples between successive frames
            frame_length: Length of each frame for analysis
        """
        self.hop_length = hop_length
        self.frame_length = frame_length

        # Musical note mapping
        self.note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def hz_to_note(self, frequency):
        """
        Convert frequency in Hz to musical note

        Args:
            frequency: Frequency in Hz

        Returns:
            Tuple of (note_name, octave, cents_deviation)
        """
        if frequency <= 0:
            return None, None, None

        # A4 = 440 Hz is our reference
        A4 = 440.0

        # Calculate the number of semitones from A4
        semitones_from_A4 = 12 * np.log2(frequency / A4)

        # Calculate note index (0 = C, 1 = C#, etc.)
        note_index = int(round(semitones_from_A4)) % 12
        note_index = (note_index + 9) % 12  # Adjust so A=9, then shift to C=0

        # Calculate octave
        octave = int((semitones_from_A4 + 9) / 12) + 4

        # Calculate cents deviation from perfect pitch
        perfect_semitones = round(semitones_from_A4)
        cents = (semitones_from_A4 - perfect_semitones) * 100

        return self.note_names[note_index], octave, cents

    def extract_pitch_sequence(self, audio_file, confidence_threshold=0.3):
        """
        Extract pitch sequence from audio file

        Args:
            audio_file: Path to audio file
            confidence_threshold: Minimum confidence for pitch detection

        Returns:
            Dictionary with pitch data
        """
        print(f"🎵 Extracting pitch from: {os.path.basename(audio_file)}")

        try:
            # Load audio
            y, sr = librosa.load(audio_file)

            # Extract pitch using piptrack (more robust than yin for polyphonic audio)
            pitches, magnitudes = librosa.piptrack(
                y=y, sr=sr, hop_length=self.hop_length, fmin=80, fmax=2000
            )

            # Get the most prominent pitch at each time frame
            pitch_sequence = []
            magnitude_sequence = []

            for t in range(pitches.shape[1]):
                # Find the frequency bin with highest magnitude
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                magnitude = magnitudes[index, t]

                # Only keep pitches above confidence threshold
                if magnitude > confidence_threshold and pitch > 0:
                    pitch_sequence.append(pitch)
                    magnitude_sequence.append(magnitude)
                else:
                    pitch_sequence.append(0)  # No clear pitch detected
                    magnitude_sequence.append(0)

            # Convert to time array
            times = librosa.frames_to_time(
                np.arange(len(pitch_sequence)), sr=sr, hop_length=self.hop_length
            )

            # Convert pitches to notes
            note_sequence = []
            for pitch in pitch_sequence:
                if pitch > 0:
                    note, octave, cents = self.hz_to_note(pitch)
                    if note and octave:
                        note_sequence.append(f"{note}{octave}")
                    else:
                        note_sequence.append("None")
                else:
                    note_sequence.append("None")

            print(f"✓ Extracted {len(pitch_sequence)} pitch frames")

            return {
                "times": times,
                "pitches": np.array(pitch_sequence),
                "magnitudes": np.array(magnitude_sequence),
                "notes": note_sequence,
                "sample_rate": sr,
            }

        except Exception as e:
            print(f"❌ Error extracting pitch: {e}")
            return None

    def find_continuous_same_notes(
        self, pitch_data1, pitch_data2, min_duration=0.5, note_tolerance=50
    ):
        """
        Find continuous segments where both audio files play the same notes

        Args:
            pitch_data1, pitch_data2: Pitch data from both files
            min_duration: Minimum duration (seconds) for a matching segment
            note_tolerance: Tolerance in cents for pitch matching

        Returns:
            List of matching segments
        """
        print("🔍 Analyzing continuous same note segments...")

        # Synchronize time arrays (use the shorter one as reference)
        min_length = min(len(pitch_data1["times"]), len(pitch_data2["times"]))

        times1 = pitch_data1["times"][:min_length]
        times2 = pitch_data2["times"][:min_length]
        notes1 = pitch_data1["notes"][:min_length]
        notes2 = pitch_data2["notes"][:min_length]
        pitches1 = pitch_data1["pitches"][:min_length]
        pitches2 = pitch_data2["pitches"][:min_length]

        matching_segments = []
        current_segment = None

        for i in range(min_length):
            # Check if both notes are valid (not "None")
            if notes1[i] != "None" and notes2[i] != "None":
                # Check if notes are the same
                same_note = notes1[i] == notes2[i]

                # Alternative: check if pitches are close enough (in Hz)
                if not same_note and pitches1[i] > 0 and pitches2[i] > 0:
                    # Calculate cents difference
                    cents_diff = abs(1200 * np.log2(pitches1[i] / pitches2[i]))
                    same_note = cents_diff < note_tolerance

                if same_note:
                    if current_segment is None:
                        # Start new segment
                        current_segment = {
                            "start_time": times1[i],
                            "start_index": i,
                            "note": notes1[i],
                            "pitch1": pitches1[i],
                            "pitch2": pitches2[i],
                        }
                    else:
                        # Continue current segment if same note
                        if notes1[i] == current_segment["note"] or (
                            pitches1[i] > 0
                            and pitches2[i] > 0
                            and abs(1200 * np.log2(pitches1[i] / current_segment["pitch1"]))
                            < note_tolerance
                        ):
                            continue
                        else:
                            # Note changed, close current segment and start new one
                            duration = times1[i - 1] - current_segment["start_time"]
                            if duration >= min_duration:
                                current_segment["end_time"] = times1[i - 1]
                                current_segment["end_index"] = i - 1
                                current_segment["duration"] = duration
                                matching_segments.append(current_segment)

                            # Start new segment
                            current_segment = {
                                "start_time": times1[i],
                                "start_index": i,
                                "note": notes1[i],
                                "pitch1": pitches1[i],
                                "pitch2": pitches2[i],
                            }
                else:
                    # No match, close current segment if exists
                    if current_segment is not None:
                        duration = times1[i - 1] - current_segment["start_time"]
                        if duration >= min_duration:
                            current_segment["end_time"] = times1[i - 1]
                            current_segment["end_index"] = i - 1
                            current_segment["duration"] = duration
                            matching_segments.append(current_segment)
                        current_segment = None
            else:
                # One or both notes are invalid, close current segment
                if current_segment is not None:
                    duration = times1[i - 1] - current_segment["start_time"]
                    if duration >= min_duration:
                        current_segment["end_time"] = times1[i - 1]
                        current_segment["end_index"] = i - 1
                        current_segment["duration"] = duration
                        matching_segments.append(current_segment)
                    current_segment = None

        # Close final segment if exists
        if current_segment is not None:
            duration = times1[-1] - current_segment["start_time"]
            if duration >= min_duration:
                current_segment["end_time"] = times1[-1]
                current_segment["end_index"] = min_length - 1
                current_segment["duration"] = duration
                matching_segments.append(current_segment)

        print(f"✓ Found {len(matching_segments)} continuous matching segments")
        return matching_segments

    def analyze_same_notes(self, file1, file2, min_duration=0.5, note_tolerance=50):
        """
        Main analysis function to find continuous same notes between two audio files

        Args:
            file1, file2: Paths to audio files
            min_duration: Minimum duration for matching segments
            note_tolerance: Tolerance in cents for pitch matching

        Returns:
            Analysis results
        """
        print("=" * 60)
        print("CONTINUOUS SAME NOTE ANALYSIS")
        print("=" * 60)

        # Extract pitch sequences
        pitch_data1 = self.extract_pitch_sequence(file1)
        pitch_data2 = self.extract_pitch_sequence(file2)

        if pitch_data1 is None or pitch_data2 is None:
            print("❌ Failed to extract pitch data from one or both files")
            return None

        # Find matching segments
        matching_segments = self.find_continuous_same_notes(
            pitch_data1, pitch_data2, min_duration, note_tolerance
        )

        # Calculate statistics
        total_duration = sum(segment["duration"] for segment in matching_segments)
        file1_duration = len(pitch_data1["times"]) * self.hop_length / pitch_data1["sample_rate"]
        file2_duration = len(pitch_data2["times"]) * self.hop_length / pitch_data2["sample_rate"]
        avg_file_duration = (file1_duration + file2_duration) / 2

        coverage_percentage = (
            (total_duration / avg_file_duration) * 100 if avg_file_duration > 0 else 0
        )

        # Display results
        print("\n📊 SAME NOTE ANALYSIS RESULTS:")
        print("-" * 40)
        print(f"Total matching segments: {len(matching_segments)}")
        print(f"Total matching duration: {total_duration:.2f} seconds")
        print(f"Coverage percentage: {coverage_percentage:.1f}%")

        if matching_segments:
            print(
                f"Longest matching segment: {max(segment['duration'] for segment in matching_segments):.2f}s"
            )
            print(f"Average segment duration: {total_duration / len(matching_segments):.2f}s")

            # Show top matching segments
            print("\n🎵 TOP MATCHING SEGMENTS:")
            print("-" * 40)
            sorted_segments = sorted(matching_segments, key=lambda x: x["duration"], reverse=True)
            for i, segment in enumerate(sorted_segments[:5]):
                print(
                    f"{i + 1}. Note {segment['note']} - "
                    f"{segment['start_time']:.1f}s to {segment['end_time']:.1f}s "
                    f"({segment['duration']:.2f}s)"
                )

        return {
            "segments": matching_segments,
            "total_duration": total_duration,
            "coverage_percentage": coverage_percentage,
            "pitch_data1": pitch_data1,
            "pitch_data2": pitch_data2,
        }

    def visualize_same_notes(self, analysis_results, file1, file2):
        """
        Create visualization of same note analysis

        Args:
            analysis_results: Results from analyze_same_notes
            file1, file2: Original file paths for labeling
        """
        if not analysis_results or not analysis_results["segments"]:
            print("❌ No data to visualize")
            return

        segments = analysis_results["segments"]
        pitch_data1 = analysis_results["pitch_data1"]
        pitch_data2 = analysis_results["pitch_data2"]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))

        # Plot 1: Pitch over time for both files
        ax1.plot(
            pitch_data1["times"],
            pitch_data1["pitches"],
            label=f"File 1: {os.path.basename(file1)}",
            alpha=0.7,
            linewidth=1,
        )
        ax2.plot(
            pitch_data2["times"],
            pitch_data2["pitches"],
            label=f"File 2: {os.path.basename(file2)}",
            alpha=0.7,
            linewidth=1,
            color="orange",
        )

        # Highlight matching segments
        for segment in segments:
            ax1.axvspan(
                segment["start_time"],
                segment["end_time"],
                alpha=0.3,
                color="green",
                label="Matching Segment" if segment == segments[0] else "",
            )
            ax2.axvspan(
                segment["start_time"],
                segment["end_time"],
                alpha=0.3,
                color="green",
                label="Matching Segment" if segment == segments[0] else "",
            )

        ax1.set_ylabel("Pitch (Hz)")
        ax1.set_title(f"Pitch Analysis: {os.path.basename(file1)}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_ylabel("Pitch (Hz)")
        ax2.set_title(f"Pitch Analysis: {os.path.basename(file2)}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Matching segments timeline
        for i, segment in enumerate(segments):
            ax3.barh(
                i,
                segment["duration"],
                left=segment["start_time"],
                alpha=0.7,
                label=f"Note {segment['note']}",
            )
            # Add text annotation
            ax3.text(
                segment["start_time"] + segment["duration"] / 2,
                i,
                f"{segment['note']}",
                ha="center",
                va="center",
                fontweight="bold",
            )

        ax3.set_xlabel("Time (seconds)")
        ax3.set_ylabel("Segment Index")
        ax3.set_title("Continuous Same Note Segments")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save visualization
        output_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "same_notes_analysis.png"
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"\n📊 Visualization saved as: {output_file}")
        plt.show()


def main():
    """Main function to run same note analysis"""
    # Initialize detector
    detector = SameNoteDetector()

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

    print("🎵 Analyzing continuous same notes between:")
    print(f"   File 1: {os.path.basename(file1)}")
    print(f"   File 2: {os.path.basename(file2)}")

    # Perform analysis
    results = detector.analyze_same_notes(file1, file2, min_duration=0.3, note_tolerance=100)

    if results:
        # Create visualization
        detector.visualize_same_notes(results, file1, file2)

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Found {len(results['segments'])} continuous matching segments")
        print(f"Total matching time: {results['total_duration']:.2f} seconds")
        print(f"Coverage: {results['coverage_percentage']:.1f}% of average song duration")


if __name__ == "__main__":
    main()
