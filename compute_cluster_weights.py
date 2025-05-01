#!/usr/bin/env python
import json
import argparse
import collections

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

def load_cluster_ids(manifest_path):
    """Read cluster_id from each line of a JSONL manifest."""
    ids = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            cid = entry.get("cluster_id")
            if cid is None or cid < 0:
                continue
            ids.append(cid)
    return np.array(ids, dtype=np.int64)

def compute_weights(ids, method="sklearn"):
    """Compute per-class weights from an array of integer labels."""
    classes = np.unique(ids)
    if method == "sklearn":
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=ids
        )
    else:
        # simple inverse‐frequency: w_k = N_total / (K * count_k)
        counts = collections.Counter(ids)
        N = len(ids)
        K = len(classes)
        weights = np.array([N / (K * counts[k]) for k in classes], dtype=np.float32)
    return classes, weights

def main():
    parser = argparse.ArgumentParser(
        description="Compute class weights from a cluster‐labeled manifest"
    )
    parser.add_argument("--manifest",
                        default="data/syllabert_clean100/clustering/labeled_manifest.jsonl",
                        help="Path to JSONL manifest with a `cluster_id` field")
    parser.add_argument("--output", 
                        default="data/syllabert_clean100/clustering/cluster_weights.pt",
                        help="Where to save weights (.pt). If omitted, prints to stdout.")
    parser.add_argument("--method", choices=["sklearn","inverse"], default="sklearn",
                        help="Use sklearn.compute_class_weight or simple inverse frequency")
    args = parser.parse_args()

    ids = load_cluster_ids(args.manifest)
    classes, weights = compute_weights(ids, method=args.method)

    # Build a full-size weight vector (assuming class labels are contiguous 0..K-1)
    K = classes.max() + 1
    full_weights = np.ones(K, dtype=np.float32)
    full_weights[classes] = weights

    weight_tensor = torch.from_numpy(full_weights)

    if args.output:
        torch.save(weight_tensor, args.output)
        print(f"Saved weight tensor (size {weight_tensor.numel()}) to {args.output}")
    else:
        print("Class weights:")
        for k, w in enumerate(weight_tensor.tolist()):
            print(f"  class {k}: {w:.4f}")

if __name__=="__main__":
    main()