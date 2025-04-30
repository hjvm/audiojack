import glob
import librosa
import numpy as np
from sklearn.cluster import KMeans
import joblib
import os
import json
from collections import Counter
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# 1. Gather 39‐dim MFCCs and metadata
all_feats = []
metadata = []  # list of tuples (wav_path, utt_id, segment_index, start, end)
sr = 16000
mfcc_hop = 0.010  # 10ms
mfcc_win = 0.025  # 25ms

librispeech_root = "data/LibriSpeech/train-clean-100"
flac_files = sorted(glob.glob(os.path.join(librispeech_root, "**/*.flac"), recursive=True))
for wav_path in tqdm(flac_files, desc="Processing LibriSpeech"):
    y, _ = librosa.load(wav_path, sr=sr)
    # compute 13 static MFCCs
    mfcc_raw = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13,
                                    hop_length=int(sr*mfcc_hop),
                                    n_fft=int(sr*mfcc_win))
    # deltas
    delta  = librosa.feature.delta(mfcc_raw, order=1)
    delta2 = librosa.feature.delta(mfcc_raw, order=2)
    mfcc_all = np.vstack([mfcc_raw, delta, delta2]).T  # (T100,39)
    # downsample to 50Hz
    T100 = mfcc_all.shape[0]
    n_pairs = T100 // 2
    mfcc50 = mfcc_all[:n_pairs*2].reshape(n_pairs, 2, -1).mean(axis=1)  # (n_pairs,39)
    # record features and metadata
    utt_id = os.path.splitext(os.path.basename(wav_path))[0]
    for idx, feat in enumerate(mfcc50):
        start = idx * 0.02
        end   = start + 0.02
        all_feats.append(feat)
        metadata.append((wav_path, utt_id, idx, start, end))

# convert to array and save
all_feats = np.vstack(all_feats)  # (N_total_frames,39)
np.save("data/hubert_clean100/clustering/hidden_frames_mfcc50_39.npy", all_feats)

# Standardize the vectors
print("Standardizing MFCC vectors...")
all_feats = StandardScaler().fit_transform(all_feats)

# 2. Fit KMeans
print(f"Running k-means with k=100 on {all_feats.shape[0]} standard-scaled MFCC50 vectors...")
n_clusters = 100
kmeans = KMeans(n_clusters=n_clusters, random_state=0, verbose=1).fit(all_feats)
joblib.dump(kmeans, "data/hubert_clean100/clustering/cluster_mfcc50_100.pkl")

# 3. Write labeled manifest
output_manifest = "data/hubert_clean100/clustering/frame_labeled_manifest.jsonl"
with open(output_manifest, "w", encoding="utf-8") as fout:
    labels = kmeans.predict(all_feats)
    for (wav_path, utt_id, idx, start, end), lab in tqdm(zip(metadata, labels), desc="Writing manifest"):
        entry = {
            "audio_file": os.path.abspath(wav_path),
            "utterance_id": utt_id,
            "segment_index": idx,
            "segment_start": round(start, 6),
            "segment_end": round(end, 6),
            "duration": round(end - start, 6),
            "cluster_id": int(lab)
        }
        fout.write(json.dumps(entry) + "\n")
print(f"Saved frame-labeled manifest to {output_manifest}")
