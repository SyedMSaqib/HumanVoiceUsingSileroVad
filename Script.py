import sys
import numpy as np
import torch
import librosa
from pydub import AudioSegment
import noisereduce as nr
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def extract_audio(video_path: str) -> tuple:
    try:
        audio = AudioSegment.from_file(video_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        rms = np.sqrt(np.mean(samples**2))
        samples = samples / (rms + 1e-8) * 0.1
        reduced_noise = nr.reduce_noise(
            y=samples,
            sr=16000,
            stationary=False,
            prop_decrease=0.75,
            n_fft=1024,
            hop_length=256,
        )
        S = librosa.stft(reduced_noise, n_fft=1024)
        freq_bins = librosa.fft_frequencies(sr=16000, n_fft=1024)
        mask = (freq_bins >= 300) & (freq_bins <= 3400)
        S_filtered = S * mask[:, np.newaxis]
        filtered_audio = librosa.istft(S_filtered)
        return filtered_audio, 16000
    except Exception as e:
        logging.error(f"Audio processing failed: {str(e)}")
        sys.exit(1)


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 script.py video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    try:
        audio_samples, sample_rate = extract_audio(video_path)
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            verbose=False,
        )
        model.eval()
        get_speech_timestamps = utils[0]
        audio_tensor = torch.from_numpy(audio_samples).float()
        segments = get_speech_timestamps(
            audio_tensor,
            model,
            sampling_rate=sample_rate,
            threshold=0.45,
            min_speech_duration_ms=300,
            speech_pad_ms=150,
            window_size_samples=512,
        )
        merged_segments = []
        for seg in segments:
            if (
                merged_segments
                and (seg["start"] - merged_segments[-1]["end"]) < 0.5 * sample_rate
            ):
                merged_segments[-1]["end"] = seg["end"]
            else:
                merged_segments.append(seg)
        total_duration = len(audio_samples) / sample_rate
        speech_duration = (
            sum((s["end"] - s["start"]) for s in merged_segments) / sample_rate
        )
        speech_percent = (speech_duration / total_duration) * 100
        print(f"\nVideo Duration: {total_duration:.1f}s")
        print(f"Speech Detected: {speech_percent:.1f}% ({speech_duration:.1f}s)")
        print(f"Speech Segments: {len(merged_segments)}")
        if merged_segments:
            print("\nSpeech Segments (start -> end):")
            for seg in merged_segments:
                start = seg["start"] / sample_rate
                end = seg["end"] / sample_rate
                print(f"- {start:.1f}s to {end:.1f}s")
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
