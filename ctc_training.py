import argparse, json, torch, os
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import List, Dict, Any

from transformers import (
    Wav2Vec2CTCTokenizer,
    HubertForCTC,
    HubertConfig,
    TrainingArguments,
    Trainer,
)

from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer

from ctc_dataset import SyllableDataset

import evaluate
import torch
from itertools import zip_longest


SAMPLE_RATE, FRAME_STRIDE = 16_000, 320


def load_vocab(vocab_path: str) -> Dict[str, int]:
    with open(vocab_path) as fp:
        return json.load(fp)

def create_tokenizer(vocab: Dict[str, int], save_dir: str) -> Wav2Vec2CTCTokenizer:
    os.makedirs(save_dir, exist_ok=True)
    vocab_file = os.path.join(save_dir, "vocab.json")
    with open(vocab_file, "w") as fp:
        json.dump(vocab, fp)
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file,
        pad_token="<pad>",
        unk_token="<unk>",
        do_lower_case=False,
        word_delimiter_token="|",   # not used but must be set
    )
    tokenizer.save_pretrained(save_dir)
    return tokenizer


@dataclass
class DataCollatorCTC:
    processor: Wav2Vec2Processor
    pad_token_id: int = 0          # <pad> id in your vocab

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        unpad_wavs = [item["waveform"].squeeze().numpy() for item in batch] 


        batch_inputs = self.processor.feature_extractor(
            unpad_wavs,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,           # pad to longest in batch
        )

        # -------- labels --------
        labels = [item["labels"] for item in batch]        
        max_len = max(x.size(0) for x in labels)
        padded = torch.full((len(labels), max_len), self.pad_token_id, dtype=torch.long)
        for i, lab in enumerate(labels):
            padded[i, : lab.size(0)] = lab
        # Transformer‑CTC expects −100 where we want to ignore loss
        padded[padded == self.pad_token_id] = -100
        batch_inputs["labels"] = padded
        return batch_inputs



def main(args):

    vocab = load_vocab(args.vocab)
    
    train_ds = SyllableDataset(data = args.manifest)
    valid_ds = SyllableDataset(data = args.valid)


    base_name = "facebook/hubert-base-ls960"
    # processor = AutoProcessor.from_pretrained(base_name, token = "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")
    # processor = AutoProcessor.from_pretrained(base_name)
    tokenizer = Wav2Vec2CTCTokenizer(args.vocab, unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    collator = DataCollatorCTC(processor, pad_token_id=vocab["<pad>"])
    config = HubertConfig.from_pretrained(base_name, vocab_size=len(vocab))
    model = HubertForCTC.from_pretrained(base_name, config=config)
    model.freeze_feature_extractor()                # often speeds convergence


    targs = TrainingArguments(
        output_dir="./hubert-ctc-syllable",
        per_device_train_batch_size=8,
        eval_strategy="steps",
        eval_steps=1000,
        logging_steps=100,
        save_steps=5000,
        learning_rate=3e-5,
        warmup_steps=500,
        num_train_epochs=30,
        fp16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        logging_dir="./logs",              
        logging_strategy="steps"
    )

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_logits = torch.tensor(pred.predictions)
        pred_ids = torch.argmax(pred_logits, dim=-1)
        label_ids = torch.tensor(pred.label_ids)

        # Decode predictions using CTC collapse
        pred_strs = processor.batch_decode(pred_ids, skip_special_tokens=True)

        # Decode label_ids by filtering out -100 and converting to tokens
        label_strs = []
        for label in label_ids:
            label = label[label != -100].tolist()
            label_str = processor.tokenizer.decode(label, skip_special_tokens=True)
            label_strs.append(label_str)


        wer = wer_metric.compute(predictions=pred_strs, references=label_strs)

        return {
            "syllable_error_rate": wer
        }


    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        tokenizer=processor.feature_extractor,   # for saving
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model("./hubert-ctc-syllable")
    processor.save_pretrained("./hubert-ctc-syllable")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", help="syllable vocab JSON", default='data/syllabert_phone/phone_vocab.json')
    p.add_argument("--manifest", help="train manifest (jsonl)", default='data/syllabert_phone/phone_manifest_train.jsonl')
    p.add_argument("--valid", help="valid manifest (jsonl)", default='data/syllabert_phone/phone_manifest_val.jsonl')
    main(p.parse_args())
    # python train_hubert_ctc.py --train manifest_train.jsonl --valid manifest_valid.jsonl