# audiojack
CIS 6200 - Advanced Deep Learning final project

---
# SyllaBERT: Syllable-Aware HuBERT Pretraining

SyllaBERT is a syllable-aware variant of HuBERT that operates on raw audio and predicts discrete pseudo-labels over syllables rather than fixed frames. This enables more linguistically grounded modeling of spoken language with better data and parameter efficiency.

---

## Pipeline Overview

1. **Syllable Segmentation:** Detect syllables in raw speech using a valley-peak-valley heuristic over a modulation-based amplitude envelope  
2. **MFCC Extraction:** Extract a single MFCC vector per syllable  
3. **Clustering:** Apply k-means to the MFCCs to generate pseudo-labels  
4. **Training:** Train a HuBERT-style model using raw audio input and masked syllable-label prediction

---

## Environment Setup

```bash
conda create -n syllabert python=3.11
conda activate syllabert

pip install torch torchaudio librosa tqdm scipy scikit-learn praatio
```

---
## 1. Clone the repo.
```bash
git clone git@github.com:hjvm/audiojack.git
cd audiojack
mkdir data
mkdir checkpoints
```

## 2. Download LibriSpeech 100h Subset
Download the cleaned 100h training subset of LibriSpeech to the `data` directory in `audiojack`.
```bash
cd data
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xvzf train-clean-100.tar.gz
rm -rf train-clean-100.tar.gz
```

---

## 2. Segment Audio and Extract MFCCs
Segment the training data into syllables and generate a manifest file using the `segment_syll_librispeech.py` script.  The default arguments are shown, but not required in the script call.
```bash
python segment_sylls_librispeech.py \
  --librispeech_root data/LibriSpeech/train-clean-100 \
  --output_dir data/syllabert_clean100/features \
  --manifest_file data/syllabert_clean100/features/manifest.jsonl
  --n_mfcc 13
  --samplerate 16000
```

This script:
- Segments each utterance into syllables using an amplitude-based method
- Computes an averaged 39-dimensional MFCC vector per syllable using 13 MFCCs, 13 deltas and 13 delta-deltas
- Saves `.npy` feature files and a `manifest.jsonl` describing them

---

## 3. Cluster Syllable MFCCs
Assign pseudo-labels (clustering IDs) to the extracted MFCC vectors with k-means clustering.  The default arguments are shown, but not required in the script call.
```bash
python cluster_syllable_vectors.py \        
  --vectors_file data/syllabert_clean100/features/syllable_mfcc_vectors.npy \
  --manifest_file data/syllabert_clean100/features/manifest.jsonl \
  --output_dir ./data/syllabert_clean100/clustering/ \
  --n_clusters 100
```

This script:
- Loads all MFCC vectors from the manifest
- Applies k-means clustering with `n_clusters`
- Writes a new manifest with cluster IDs as pseudo-labels

Output:
```
./data/syllabert_clean100/clustering/labeled_manifest.jsonl
./data/syllabert_clean100/clustering/syllable_cluster_labels.npy
./data/syllabert_clean100/clustering/syllable_kmeans_k100.pkl
```

---

## 4. Train the SyllaBERT Model

```bash
python train_syllabert.py \
    --manifest data/syllabert_clean100/clustering/labeled_manifest.jsonl \
    --sr 16000 \
    --hubert_model facebook/hubert-base-ls960 \
    --bs 8 \
    --lr 1e-4 \
    --epochs 35 \
    --out_dir 'checkpoints'
```

This script:
- Loads raw waveforms and syllable boundary metadata
- Applies a convolutional frontend + transformer encoder
- Uses HuBERT-style masking (65% of syllables per batch)
- Computes loss only over masked syllables
- Logs batch accuracy and loss
- Saves model checkpoints after each epoch:
  ```
  checkpoints/syllabert_wave_epochN.pt
  checkpoints/syllabert_wave_latest.pt
  ```
---
## 5. Recluster the data using SyllaBERT embeddings.

```bash
python recluster_states_syllabert.py \
    --manifest data/syllabert_clean100/hidden_clusters/labeled_manifest.jsonl \
    --sr 16000 \
    --output_dir data/syllabert_clean100/hidden_clusters \
    --checkpoint checkpoints/syllabert_latest.pt \
    --hubert_model facebook/hubert-base-ls960 \
    --librispeech_root data/LibriSpeech/train-clean-100 \
    --n_clusters 500
```
---
## 6. Cluster frame-based MFCCs (baseline)
Generate pseudo-labels for frame-based MFCC clusters as a baseline comparison.

```bash
python cluster_frame_vectors.py 
```

---
## 7. Recluster the data using HuBERT embeddings (baseline).

```bash
python recluster_states_hubert.py \
    --audio_glob data/LibriSpeech/train-clean-100/**/*.flac \
    --output_dir data/hubert_clean100/hidden_clusters \
    --librispeech_root data/LibriSpeech/train-clean-100 \
    --batch_size 10000 \
    --n_clusters 500
```
---
## 8. Evaluate clustering purity of all 4 methods.
Run through `eval_cluster_purity.ipynb`.

---
## Directory Layout

```
data/
└── syllabert_clean100/
    ├── features/
    │   ├── <utterance_id>.npy          # MFCC vectors
    │   └── manifest.jsonl              # Metadata (audio paths, timings, features)
    └── clustering/
        ├── kmeans_100.pkl              # KMeans model
        └── labeled_manifest.jsonl      # MFCCs with cluster labels
```

---

## Notes

- Syllables are pooled from raw waveforms; targets are cluster IDs over syllable MFCCs.
- The architecture matches HuBERT, but the learning unit is a syllable.
- Masking and loss computation follow HuBERT’s 65% random span masking strategy — adapted for syllables.

---

## References

- Hsu et al., 2021. [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447)
