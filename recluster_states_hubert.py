# extract_and_cluster_hubert_frames.py

import os
import glob
import torch
import librosa
import numpy as np
import joblib
import json
from collections import Counter
from transformers import HubertModel
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

def extract_hidden_frames(model, wav_paths, sr=16000, device="cpu"):
    """
    Runs each file through HuBERT and collects last_hidden_state.
    Returns a list of (num_frames, hidden_size) arrays.
    """
    feats = []
    for wav_path in tqdm(wav_paths, desc="Extracting HuBERT frames"):
        # load waveform
        y, _ = librosa.load(wav_path, sr=sr)
        x = torch.from_numpy(y).float().unsqueeze(0).to(device)  # (1, T)
        with torch.no_grad():
            out = model(x, attention_mask=None)
        # (1, T_frames, H) → (T_frames, H)
        h = out.last_hidden_state.squeeze(0).cpu().numpy()
        feats.append(h)
    return feats

def cluster_hidden_frames(frame_feats, n_clusters=100, batch_size=10000, output_path=None):
    """
    Clusters all frame_feats (list of [n_i x H] arrays) via MiniBatchKMeans.
    Saves the fitted model if output_path is provided.
    """
    H = frame_feats[0].shape[1]
    mbk = MiniBatchKMeans(n_clusters=n_clusters,
                          batch_size=batch_size,
                          random_state=0)
    # feed in mini‐batches
    for feat in tqdm(frame_feats, desc="Clustering frames"):
        # you can also sub-sample: feat[np.random.choice(len(feat), min(len(feat),batch_size),False)]
        mbk.partial_fit(feat)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(mbk, output_path)
        print(f"Saved HuBERT frame k-means to {output_path}")
    return mbk

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract HuBERT frame embeddings and cluster them"
    )
    parser.add_argument(
        "--audio_glob", type=str,
        default="data/LibriSpeech/train-clean-100/**/*.flac",
        help="glob pattern for LibriSpeech wav files"
    )
    parser.add_argument(
        "--n_clusters", type=int, default=500,
        help="number of clusters for MiniBatchKMeans"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10000,
        help="mini‐batch size for clustering"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="data/hubert_clean100/hidden_clusters",
        help="where to save the clustering model and extracted features"
    )
    args = parser.parse_args()

    # 1) Load HuBERT
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")\
                       .to(args.device).eval()

    # 2) Gather all file paths
    wav_paths = sorted(glob.glob(args.audio_glob, recursive=True))
    print(f"Found {len(wav_paths)} utterances")

    # 3) Extract hidden‐frame embeddings
    frame_feats = extract_hidden_frames(hubert, wav_paths,
                                        sr=16000, device=args.device)
    total_frames = sum([f.shape[0] for f in frame_feats])
    print(f"Extracted {total_frames} frames from {len(frame_feats)} flac files.")
    
    # Save the extracted hidden states.
    #all_feats = np.concatenate(frame_feats, axis=0)  # (num_frames, hidden_size) array
    #vectors_path = os.path.join(args.output_dir, "hidden_states_layer6.npy")
    #os.makedirs(args.output_dir, exist_ok=True)
    #print(f"Saving {all_feats.shape[0]} hidden states to {vectors_path}")
    # Save the hidden states
    #np.save(vectors_path, all_feats)

    # 4) Cluster with MiniBatchKMeans
    kmeans_model_file = os.path.join(args.output_dir, f"frame_kmeans_mfcc50_k{args.n_clusters}.pkl")
    if not os.path.isfile(kmeans_model_file):
        mbk = cluster_hidden_frames(frame_feats,
                                    n_clusters=args.n_clusters,
                                    batch_size=args.batch_size,
                                    output_path=kmeans_model_file)
    else:
        mbk = joblib.load(kmeans_model_file)
    
    # 5) Write frame-level labeled manifest using the trained KMeans
    output_manifest = os.path.join(args.output_dir, f"hidden_frame_labeled_manifest_k{args.n_clusters}.jsonl")
    conv_stride = 320  # frame hop in samples (20ms at 16kHz)
    sr = 16000
    with open(output_manifest, "w", encoding="utf-8") as fout:
        for wav_path, feats in zip(wav_paths, frame_feats):
            # Predict cluster for each frame embedding
            labels = mbk.predict(feats)
            utt_id = os.path.splitext(os.path.basename(wav_path))[0]
            for idx, lab in enumerate(labels):
                start = idx * conv_stride / sr
                end = start + conv_stride / sr
                entry = {
                    "audio_file": os.path.abspath(wav_path),
                    "utterance_id": utt_id,
                    "segment_index": idx,
                    "segment_start": round(start, 6),
                    "segment_end":   round(end, 6),
                    "duration":      round(end - start, 6),
                    "cluster_id":    int(lab)
                }
                fout.write(json.dumps(entry) + "\n")
    print(f"Saved frame-labeled manifest to {output_manifest}")


