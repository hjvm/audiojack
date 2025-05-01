#!/usr/bin/env python
"""
train_syllabert.py

Fine‑tune SyllaBERTEncoder (imported as SyllaBERT) on syllable cluster targets.
"""
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from syllabert_model import SyllaBERT
from syllable_dataset import SyllableDataset, collate_syllable_utterances
from tqdm import tqdm
import argparse
from datetime import datetime
from loguru import logger
import os
import re
from torch import nn

import warnings
warnings.filterwarnings("ignore")

now = datetime.now().strftime("%Y-%m-%d_%H-%M")

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO")
os.makedirs('./logs', exist_ok=True)
logger.add(f"logs/train_{now}.log", level="INFO", format="{time} | {level} | {message}")

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


def load_latest_checkpoint(checkpoint_dir):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if re.match(r'syllabert_epoch(\d+)\.pt', f)]
    epochs = [int(re.search(r'epoch(\d+)', f).group(1)) for f in checkpoint_files]
    latest_epoch = max(epochs)
    latest_ckpt = f"syllabert_epoch{latest_epoch}.pt"
    path = os.path.join(checkpoint_dir, latest_ckpt)
    return path, latest_epoch


def has_nan_or_inf(model):
    for param in model.parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                return True
    return False
    
def train(args):
    device = get_device()
    # instantiate model: only hubert_pretrained_model arg
    model = SyllaBERT(
        hubert_pretrained_model=args.hubert_model,
        num_classes=100  # set to number of clusters
    ).to(device)

    epoch_n = 0
    # continue training
    if args.c:
        path, epoch_n = load_latest_checkpoint('./checkpoints/')
        model.load_state_dict(torch.load(path))
        logger.info(f"Loaded checkpoint: {path}")

    # prepare data
    dataset = SyllableDataset(
        manifest_path=args.manifest,
        samplerate=args.sr
    )
    loader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=True,
        collate_fn=collate_syllable_utterances,
        pin_memory=True
    )

    num_batches = len(loader)
    logger.info(f"Number of batches: {num_batches}")

    # ================================================================================================
    # The commented code below is to freeze the convolution layers. 
    # But setting this seems to unstable the training as causes Nan loss/gradient so I commented these.
    # ================================================================================================
    for param in model.feature_extractor.parameters():
        param.requires_grad = False
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params, 
        lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01
    )
    # optimizer = torch.optim.Adam(
    #     model.parameters(), 
    #     lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01
    # )


    log_interval = 50

    for epoch in range(1 + epoch_n, args.epochs + 1 + epoch_n):
        logger.info(f"Epoch {epoch}/{args.epochs + 1 + epoch_n}")
        model.train()
        total_loss, steps = 0.0, 0
        batch_count = 0
        batch_loss = 0.
        
        with tqdm(total=num_batches, desc=f"Epoch {epoch}", dynamic_ncols=True, leave=False) as pbar:
            for batch_idx, (inputs, targets_list, segments) in enumerate(loader):
                if inputs is None:
                    continue
                # print(f'{path =}')
                # print("inputs:", inputs.shape)
                # print("targets:", targets_list.shape)

                inputs = inputs.to(device)
                
                # forward returns list of (N_syll, num_classes) logits
                logits_list = model(inputs, args.sr)

                for i, logits in enumerate(logits_list):
                    n_labels = (targets_list[i]!=-100).sum()
                    n_sylls = logits.size(0)
                    if n_labels != n_sylls:
                        n = min(n_labels, n_sylls)
                        logits_list[i] = logits[:n]
                        targets_list[i, n:] = -100

                
                
                # 1) flatten to 1D
                targets_flat = targets_list.to(device).view(-1)
                # 2) select only the non‐padded entries
                valid = targets_flat != -100
                targets = targets_flat[valid]

                # flatten and compute loss
                all_logits = torch.cat(logits_list, dim=0)

                loss = F.cross_entropy(all_logits, targets, ignore_index=-100)
                if torch.isnan(loss):
                    logger.error(f"Loss: {loss.item()}")
                    logger.error(f"Any NaN in logits: {torch.isnan(all_logits).any()}")
                    logger.error(f"Any NaN in target_tokens: {torch.isnan(targets).any()}")
                    logger.error("Loss is NaN! Skipping backward.")
                    continue




                optimizer.zero_grad()
                loss.backward()
                if has_nan_or_inf(model):
                    logger.info("NaN detected in gradients — skipping optimizer step")
                    optimizer.zero_grad()
                else:
                    optimizer.step()
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=10.0)
                    optimizer.zero_grad()

                steps += 1

                with torch.no_grad():
                    total_loss += loss.item()
                    batch_count += 1
                    batch_loss += loss.item()
                    
                    preds = all_logits.argmax(dim=-1)
                    
                    correct = (preds == targets).sum().item()
                    total = targets.size(0)
                    acc = correct / total if total > 0 else 0.0

                    if batch_idx % log_interval == 0:
                        if batch_idx != 0:
                            avg_batch_loss = batch_loss / log_interval
                        else:
                            avg_batch_loss = loss.item()
                        logger.info(
                            f"Epoch [{epoch}/{args.epochs + 1}], Batch [{batch_idx}], "
                            f"Avg Loss: {avg_batch_loss:.4f}, "
                            f"Masked Acc: {acc * 100:.2f}% ({correct}/{total})"
                        )
                        batch_loss = 0.0
                pbar.update(1)


        avg_loss = total_loss / steps if steps else 0.0
        logger.info(f"Epoch {epoch}/{args.epochs}, Loss: {avg_loss:.4f}")

        os.makedirs(args.out_dir, exist_ok=True)
        ckpt = os.path.join(args.out_dir, f"syllabert_epoch{epoch}.pt")
        model.save_checkpoint(ckpt)
        model.save_checkpoint(os.path.join(args.out_dir, "syllabert_latest.pt"))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, default='data/syllabert_clean100/clustering/labeled_manifest.jsonl')
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--hubert_model', type=str, default='facebook/hubert-base-ls960')
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--out_dir', type=str, default='checkpoints')
    parser.add_argument('-c', action='store_true', help='continue training')
    # parser.add_argument('--stage', type=int, required=True)
    args = parser.parse_args()
    train(args)
