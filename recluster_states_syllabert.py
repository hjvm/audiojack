#!/usr/bin/env python
"""
Extract and cluster the hidden states.
"""
# ================================================================================================
# I already got the extracted features of the 6th transformer, saved at:
# './data/syllabert_clean100/hidden_clusters/hidden_states_layer6.npy'
# So you don't need to run extract_hidden_states again if the model is not further trained.
#
# There some redundencies in the imports.
# ================================================================================================
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from syllabert_model import SyllaBERT
from syllable_dataset import SyllableDataset, collate_syllable_utterances
from findsylls import segment_audio
from tqdm import tqdm
import argparse
from datetime import datetime
from loguru import logger
import os
import re
import json
import torchaudio
import glob
import random
import warnings
import numpy as np
from sklearn.cluster import KMeans
import joblib
from torch import nn
warnings.filterwarnings("ignore")


now = datetime.now().strftime("%Y-%m-%d_%H-%M")
logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO")
os.makedirs('./logs', exist_ok=True)
logger.add(f"logs/reclustering_{now}.log", level="INFO", format="{time} | {level} | {message}")


# ================================================================================================
# Build the dir to store recluster data: 'data/syllabert_clean100/hidden_clusters/'
# ================================================================================================
os.makedirs("./data/syllabert_clean100/hidden_clusters", exist_ok=True)



def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


def get_utterence_id(audio_file):
    utterance_id = os.path.splitext(os.path.basename(audio_file))[0]
    return utterance_id


def extract_hidden_states(args):
    """Extract the hidden state from the 6th layer of transformers.
    """
    device = get_device()

    model = SyllaBERT(
        hubert_pretrained_model=args.hubert_model,
        num_classes=100
    ).to(device)

    model.load_checkpoint(args.checkpoint, map_location=device)
    model.eval()

    flac_files = sorted(glob.glob(os.path.join(args.librispeech_root, "**", "*.flac"), recursive=True))
    #num_samples = int(len(flac_files) * 0.1)  # random sample 10% as in the paper
    #sampled_files = random.sample(flac_files, num_samples)


    logger.info(f"Found {len(flac_files)} .flac files in {args.librispeech_root}")

    # get hidden state
    all_hidden_states = []
    for file in tqdm(flac_files, desc="Extracting hidden states"):
        waveform, _ = torchaudio.load(file)

        waveform = waveform.to(device)

        # Forward pass through the model
        with torch.no_grad():
            # ================================================================================================
            # Change hidden_layer = 6 to others if using other hidden states.
            # ================================================================================================
            hidden_states = model(waveform.unsqueeze(0), hidden_layer = 6)  # layer 6
            hidden_states = hidden_states.squeeze(0).cpu().numpy()
            #print(f'{hidden_states.shape = }')

        all_hidden_states.append(hidden_states)

    all_hidden_states = np.concatenate(all_hidden_states, axis = 0)
    print(f'{all_hidden_states.shape = }')
    output_path = "./data/syllabert_clean100/hidden_clusters/hidden_states_layer6.npy"
    np.save(output_path, all_hidden_states)
    logger.info(f"Saved {all_hidden_states.shape[0]} hidden states to {output_path}")


def cluster_hidden_states(args):
    vectors_file = os.path.join(args.output_dir, "hidden_states_layer6.npy")
    kmeans_model_file = os.path.join(args.output_dir, "kmeans_500.joblib")
    if not os.path.isfile(kmeans_model_file):
        # ================================================================================================
        # You really don't want to do the clustering every time...
        # I already got a fitted kmeans classifier stored at: 'data/syllabert_clean100/hidden_clusters/kmeans.joblib'
        # ================================================================================================
        all_hidden_states = np.load(vectors_file)
        n_clusters = args.n_clusters
        logger.info(f"Running k-means with k={n_clusters} on {all_hidden_states.shape[0]} syllable vectors...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto", verbose=1)
        kmeans.fit(all_hidden_states)
        joblib.dump(kmeans, kmeans_model_file)
    else:
        kmeans = joblib.load(kmeans_model_file)

    device = get_device()
    # instantiate model: only hubert_pretrained_model arg
    model = SyllaBERT(
        hubert_pretrained_model=args.hubert_model,
        num_classes=100  # set to number of clusters
    ).to(device)

    model.load_checkpoint(args.checkpoint, map_location=device)
    model.eval()

    flac_files = sorted(glob.glob(os.path.join(args.librispeech_root, "**", "*.flac"), recursive=True))

    metadata = []
    for file in tqdm(flac_files, desc="Clustering hidden states"):
        # ================================================================================================
        # I used flac_files[:10] to see the cluster quality. 
        # You can check the cluster_id of file './data/syllabert_clean100/hidden_clusters/labeled_manifest.jsonl'
        # to see what I was talking about today.
        # ================================================================================================
        utterance_id = get_utterence_id(file)
        waveform, _ = torchaudio.load(file)
        sylls, _, _ = segment_audio(file)

        waveform = waveform.to(device)
        # Forward pass through the model
        with torch.no_grad():
            hidden_states = model(waveform.unsqueeze(0), hidden_layer = 6)
            hidden_states = hidden_states.squeeze(0).cpu().numpy()
            #print(f'{hidden_states.shape = }')
        for idx, (state, syll )in enumerate(zip(hidden_states, sylls)):
            #print(state.shape)
            cluster_id = kmeans.predict(state.reshape(1, -1))
            start, _, end = syll
            duration = end - start
            # ================================================================================================
            # Used the same format as the previous manifest files so we can use the same dataset class and dataloader.
            # ================================================================================================
            metadata.append({
                "audio_file": os.path.abspath(file),
                "utterance_id": utterance_id,
                "segment_index": idx,
                "segment_start": float(start),
                "segment_end": float(end),
                "duration": duration,
                "cluster_id": int(cluster_id)
            })
    # Save the cluster labels
    labeled_manifest_path = args.manifest
    with open(labeled_manifest_path, "w", encoding="utf-8") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")
    logger.info(f"Saved manifest with metadata to {labeled_manifest_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, default='data/syllabert_clean100/hidden_clusters/labeled_manifest.jsonl')
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--output_dir', type=str, default='data/syllabert_clean100/hidden_clusters')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/syllabert_latest.pt')
    parser.add_argument('--hubert_model', type=str, default='facebook/hubert-base-ls960')
    # ================================================================================================
    # Remember to the librispeech_root to your path.
    # ================================================================================================
    parser.add_argument('--librispeech_root', type=str, default='data/LibriSpeech/train-clean-100')  # Change this default setting.
    parser.add_argument('--n_clusters', type=int, default=500)
    # parser.add_argument('--stage', type=int, required=True)
    args = parser.parse_args()

    extract_hidden_states(args)  # You can skip this if the model is not further trained.
    cluster_hidden_states(args)

