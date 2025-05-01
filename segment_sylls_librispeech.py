# segment_mean_mfcc_librispeech.py

import os
import glob
import json
import argparse
import numpy as np
import librosa
from tqdm import tqdm

from findsylls import segment_audio  # assumes your segmentation function is imported from findsylls.py

def process_librispeech_file(audio_file, output_dir, samplerate=16000, n_mfcc=13, include_delta=True, include_ddelta=True):
    utterance_id = os.path.splitext(os.path.basename(audio_file))[0]
    #speaker_id = audio_file.split("/")[-3]
    #chapter_id = audio_file.split("/")[-2]
    #utterance_id = f"{speaker_id}-{chapter_id}-{base}"
    n_fft = int(0.025 * samplerate)  # 25 ms window
    hop_length = int(0.01 * samplerate)  # 10 ms hop

    try:
        syllables, t, A = segment_audio(audio_file, samplerate=samplerate, show_plots=False)
        audio, sr = librosa.load(audio_file, sr=samplerate)
    except Exception as e:
        print(f"Skipping {audio_file} due to error: {e}")
        return []

    vectors_info = []
    mfcc_vectors = []

    for idx, (start, peak, end) in enumerate(syllables):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        syll = audio[start_sample:end_sample]
        duration = (end - start)

        if len(syll) < int(0.025 * sr):  # skip segments shorter than 25 ms
            print(f"Skipping segment {idx} in {audio_file} due to short duration: {duration:.3f}s")
            continue

        # compute static 13-dim MFCCs
        mfcc_raw   = librosa.feature.mfcc(
                        y=syll, sr=sr,
                        n_mfcc=n_mfcc,
                        n_fft=n_fft,
                        hop_length=hop_length
                    )  # shape (n_mfcc, T)
        # adapt delta width to segment length
        n_frames = mfcc_raw.shape[1]
        # desired window length for delta: default 9, but must be <= n_frames
        default_width = 9
        if n_frames < default_width:
            width = n_frames if (n_frames % 2 == 1) else max(n_frames - 1, 1)
        else:
            width = default_width
        # compute delta and delta-delta
        mfcc_vstack = [mfcc_raw]
        if include_delta:
            mfcc_vstack.append(librosa.feature.delta(mfcc_raw, order=1, width=width))
        if include_ddelta:
            mfcc_vstack.append(librosa.feature.delta(mfcc_raw, order=2, width=width))
        #mfcc_delta  = librosa.feature.delta(mfcc_raw, order=1, width=width)
        #mfcc_delta2 = librosa.feature.delta(mfcc_raw, order=2, width=width)        # stack to (3*n_mfcc, T) = (39, T) when n_mfcc=13
        if len(mfcc_vstack) > 1:
            mfcc_all = np.vstack(mfcc_vstack)  # shape (3*n_mfcc, T)
        else:
            mfcc_all = mfcc_raw
        # mean over time axis → (3*n_mfcc,)
        mfcc_mean = mfcc_all.mean(axis=1)
        mfcc_vectors.append(mfcc_mean)

        vectors_info.append({
            "audio_file": os.path.abspath(audio_file),
            "utterance_id": utterance_id,
            "segment_index": idx,
            "segment_start": float(start),
            "segment_end": float(end),
            "duration": duration,
            "vector_index": len(mfcc_vectors) - 1
        })

    return mfcc_vectors, vectors_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech_root", type=str,
                         default="data/LibriSpeech/train-clean-100",                                          
                         help="Path to LibriSpeech split root")
    parser.add_argument("--output_dir", type=str,
                        default="data/syllabert_clean100/features",
                        help="Directory to save MFCC numpy files")
    parser.add_argument("--manifest_file", type=str,
                        default="data/syllabert_clean100/features/manifest.jsonl",
                        help="Path to output manifest file (JSONL)")
    parser.add_argument("--n_mfcc", type=int, 
                        default=13, 
                        help="Number of MFCC coefficients")
    parser.add_argument("--include_delta", type=bool, 
                        default=True, 
                        help="Include first order MFCC delta coefficients")
    parser.add_argument("--include_ddelta", type=bool, 
                        default=True, 
                        help="Include second order MFCC delta coefficients")
    parser.add_argument("--samplerate", type=int, 
                        default=16000, 
                        help="Sampling rate")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    flac_files = sorted(glob.glob(os.path.join(args.librispeech_root, "**/*.flac"), recursive=True))
    print(f"Found {len(flac_files)} .flac files in {args.librispeech_root}")

    all_vectors = []
    all_metadata = []

    for file in tqdm(flac_files, desc="Processing LibriSpeech"):
        result = process_librispeech_file(
            audio_file=file,
            output_dir=args.output_dir,
            samplerate=args.samplerate,
            n_mfcc=args.n_mfcc,
            include_delta=args.include_delta,
            include_ddelta=args.include_ddelta
        ) 
        if not result:
            continue
        vectors, metadata = result
        all_vectors.extend(vectors)
        all_metadata.extend(metadata)

    # Save all MFCC mean vectors in one file
    vectors_np = np.stack(all_vectors, axis=0)
    vectors_path = os.path.join(args.output_dir, "syllable_mfcc_vectors.npy")
    np.save(vectors_path, vectors_np)

    # Save manifest metadata
    manifest_path = args.manifest_file
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in all_metadata:
            f.write(json.dumps(entry) + "\n")

    print(f"Saved {len(all_vectors)} syllable MFCC vectors to {vectors_path}")
    print(f"Saved manifest with metadata to {manifest_path}")

if __name__ == "__main__":
    main()
