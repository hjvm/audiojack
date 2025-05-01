import torch
import torch.nn as nn
import os
import numpy as np
from transformers import HubertModel, HubertConfig 
from findsylls import segment_waveform

class SyllaBERTEncoder(nn.Module):
    """HuBERT-based encoder that segments audio into syllables and
    produces a per-syllable token sequence via mean-pooling."""
    def __init__(self, hubert_pretrained_model: str = None, num_classes: int = None):
        super().__init__()
        hubert_name = hubert_pretrained_model or "facebook/hubert-base-ls960"
        # Load pretrained HuBERT model
        config = HubertConfig.from_pretrained(hubert_name)
        config.do_stable_layer_norm = True
        print(config)
        self.hubert = HubertModel(config)
        # Use HuBERT's convolutional feature extractor
        self.feature_extractor = self.hubert.feature_extractor
        # Compute total stride of conv stack for frame-to-time mapping
        self.conv_stride = int(np.prod([layer.conv.stride[0] for layer in self.feature_extractor.conv_layers]))
        # Reuse HuBERT's transformer encoder and layer norm
        self.transformer = self.hubert.encoder
        self.layer_norm = getattr(self.transformer, 'layer_norm', nn.LayerNorm(self.hubert.config.hidden_size))
        # Final projection to vocabulary logits
        out_dim = num_classes if num_classes is not None else self.hubert.config.vocab_size
        # Final projection to cluster logits
        self.projection = nn.Linear(self.hubert.config.hidden_size, out_dim)

    def forward(self, waveforms: torch.Tensor, sampling_rate: int = 16000, hidden_layer: int = None):
        """
        Args:
            waveforms: Tensor of shape (B, 1, T_samples)
            sampling_rate: sampling rate of audio in Hz
        Returns:
            List of length B, each entry Tensor of shape (n_syllables_i, vocab_size)
        """
        assert not torch.isnan(waveforms).any(), "Nan in input"

        B, _, T = waveforms.size()
        device = waveforms.device
        # 1) Extract convolutional features
        conv_in = waveforms.squeeze(1)                 # (B, T)
        feats = self.feature_extractor(conv_in)        # (B, C, T')

        assert not torch.isnan(feats).any(), "Nan in feats"
        
        feats = feats.transpose(1, 2)                  # (B, T', C)
        T_frames = feats.size(1)
        # 2) Syllable segmentation and pooling
        pooled = []
        for b in range(B):
            wav_np = waveforms[b,0].cpu().numpy()
            sylls, _, _ = segment_waveform(wav_np, sampling_rate)
            segments = []
            for start_s, _, end_s in sylls:
                s_fr = np.floor(start_s * sampling_rate / self.conv_stride)
                e_fr = np.ceil (end_s   * sampling_rate / self.conv_stride)

                # clamp to [0, T_frames] and guarantee at least 1 frame
                s_fr = int(max(0, min(s_fr, T_frames - 1)))
                e_fr = int(max(s_fr + 1, min(e_fr, T_frames)))
                segments.append((s_fr, e_fr))
            if not segments:
                reps = feats[b].mean(dim=0, keepdim=True)  # (1, C)
            else:
                reps = torch.stack([feats[b, s:e].mean(dim=0) for s,e in segments])  # (n_syll, C)

            
            pooled.append(reps)
        # 3) Pad pooled sequences and create transformer mask
        max_syll = max(p.size(0) for p in pooled)
        hidden_size = feats.size(2)

        # assert not torch.isnan(torch.tensor(pooled)).any(), "Nan in pooled"

        padded = torch.zeros(B, max_syll, hidden_size, device=device)


        padding_mask = torch.ones(B, max_syll, dtype=torch.bool, device=device)
        for b, reps in enumerate(pooled):

            assert not torch.isnan(reps).any(), "Nan in reps"

            L = reps.size(0)
            padded[b, :L] = reps
            padding_mask[b, :L] = False
        # 4) Project pooled conv features to hidden dim

        assert not torch.isnan(padded).any(), "Nan in padded after mask"

        proj = self.hubert.feature_projection(padded)  # (B, max_syll, hidden_size)
        
        assert not torch.isnan(proj).any(), "Nan in proj before layer norm"



        # 5) Build attention mask: 1 for keep, 0 for pad
        attention_mask = (~padding_mask) # dtype=torch.bool

        # 6) Configure transformer kwargs
        tf_kwargs = {
            "output_hidden_states": (hidden_layer is not None),
            "return_dict": False
        }
        if padding_mask.any():
             tf_kwargs["attention_mask"] = attention_mask

        # 7) Run HuBERT encoder on syllable embeddings with optional masking
        proj = self.layer_norm(proj)

        assert not torch.isnan(proj).any(), "Nan in proj after layer norm"

        encoder_output = self.transformer(proj, **tf_kwargs)
        # tuple format: (last_hidden_state, hidden_states, ...)
        if not tf_kwargs["output_hidden_states"]:
            x = encoder_output[0] # (B, max_syll, hidden_size)
        else:
            return encoder_output[1][hidden_layer][0]
        
        assert not torch.isnan(x).any(), "Nan in transformer"

        x = self.layer_norm(x)
        # 7) Final projection per utterance per utterance
        outputs = []
        for b in range(B):
            L = (~padding_mask[b]).sum().item()
            logits = self.projection(x[b, :L])             # (L, vocab_size)
            outputs.append(logits)
        return outputs

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: str, map_location=None):
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state)
        print(f"Loaded checkpoint from {path}")

# Alias for backward compatibility
SyllaBERT = SyllaBERTEncoder

# Optional: HuBERT-style random masking utility
def hubert_style_mask(batch_size, seq_len, mask_prob=0.065, mask_length=10):
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    num_mask = int(mask_prob * seq_len / mask_length + 0.5)
    for b in range(batch_size):
        starts = torch.randperm(seq_len - mask_length)[:num_mask]
        for st in starts:
            mask[b, st:st + mask_length] = True
    return mask
