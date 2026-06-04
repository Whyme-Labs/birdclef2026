"""V179: Silero-VAD speech stripping for train_audio.

Both BirdCLEF 2025 1st place and top-2% used Silero-VAD to remove recordist
voice ("alien speech") from focal recordings before training. Speech artifacts
are a dominant false-positive source for rare species with few recordings.

Recipe:
  1. Run Silero-VAD on each train_audio file
  2. Mask voiced regions to silence (zero out)
  3. Save to data/train_audio_devoiced/

Then retrain V2S with --data_dir pointing to devoiced version.

Silero-VAD model URL: https://github.com/snakers4/silero-vad
Loaded via torch.hub (requires network or cached model).
"""
import argparse, time
from pathlib import Path
import torch
import torchaudio
import soundfile as sf
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="data/train_audio")
    ap.add_argument("--output_dir", default="data/train_audio_devoiced")
    ap.add_argument("--threshold", type=float, default=0.5, help="VAD speech probability threshold")
    ap.add_argument("--min_speech_ms", type=int, default=250)
    ap.add_argument("--min_silence_ms", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print("Loading Silero-VAD...")
    # Try local cache first, then hub
    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
    except Exception as e:
        print(f"torch.hub failed: {e}")
        print("Try: pip install silero-vad   then load via silero_vad library directly")
        return
    (get_speech_timestamps, _, read_audio, *_) = utils

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.rglob("*.ogg"))
    if args.limit > 0:
        files = files[:args.limit]
    print(f"Files: {len(files)}")

    SR = 16000  # Silero requires 16kHz; we resample, detect, then upsample mask

    t0 = time.time()
    n_devoiced = 0
    total_voice_removed = 0.0
    total_audio = 0.0
    for fi, fp in enumerate(files):
        # Mirror directory structure
        rel = fp.relative_to(in_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            wav, sr_orig = sf.read(str(fp))
            if wav.ndim > 1: wav = wav.mean(axis=1)

            # Resample to 16k for VAD
            if sr_orig != SR:
                wav_16 = torchaudio.functional.resample(
                    torch.from_numpy(wav.astype(np.float32)), sr_orig, SR
                ).numpy()
            else:
                wav_16 = wav.astype(np.float32)

            speech_ts = get_speech_timestamps(
                torch.from_numpy(wav_16),
                model,
                threshold=args.threshold,
                sampling_rate=SR,
                min_speech_duration_ms=args.min_speech_ms,
                min_silence_duration_ms=args.min_silence_ms,
            )

            total_audio += len(wav_16) / SR
            if speech_ts:
                n_devoiced += 1
                # Build mask in original-rate samples
                wav_out = wav.astype(np.float32).copy()
                for seg in speech_ts:
                    s = int(seg["start"] * sr_orig / SR)
                    e = int(seg["end"] * sr_orig / SR)
                    s = max(0, s); e = min(len(wav_out), e)
                    wav_out[s:e] = 0.0
                    total_voice_removed += (e - s) / sr_orig
                sf.write(str(out_path), wav_out, sr_orig)
            else:
                # No speech detected — copy original
                sf.write(str(out_path), wav, sr_orig)

            if (fi + 1) % 200 == 0:
                rate = (fi + 1) / (time.time() - t0)
                print(f"  {fi+1}/{len(files)}, devoiced={n_devoiced}, "
                      f"voice%={100*total_voice_removed/total_audio:.2f}, rate={rate:.1f}/s")
        except Exception as e:
            print(f"  skip {fp.name}: {e}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print(f"Files devoiced: {n_devoiced}/{len(files)} ({100*n_devoiced/len(files):.1f}%)")
    print(f"Total audio: {total_audio:.0f}s, voice removed: {total_voice_removed:.0f}s "
          f"({100*total_voice_removed/total_audio:.2f}%)")


if __name__ == "__main__":
    main()
