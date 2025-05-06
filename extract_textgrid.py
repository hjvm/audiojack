from textgrid import TextGrid
import json
from tqdm import tqdm
import os
from collections import Counter

audio_dir = "./data/LibriSpeech/train-clean-100"
textgrid_dir = "./data/LibriSpeech_syllable"



audio_paths = []
textgrid_paths = []
for root, _, files in os.walk(audio_dir):   
    for file in files:
        if file.endswith(".flac"):
            audio_path = os.path.join(root, file)
            textgrid_path = os.path.join(
                textgrid_dir,
                os.path.relpath(audio_path, audio_dir).replace(".flac", "_syllabified.TextGrid")
            )
            if os.path.exists(textgrid_path):
                audio_paths.append(audio_path)
                textgrid_paths.append(textgrid_path)

syllable_counter = Counter()
for tg_path in tqdm(textgrid_paths):
    tg = TextGrid.fromFile(tg_path)

    for tier in tg.tiers:
        # print(tier.name)
        if "phones" in tier.name.lower():
            # syllable_tier = tier
            # break
            for interval in tier:
                
                mark = interval.mark.strip()
                if mark:
                    syllable_counter[mark] += 1

vocab = {
    "<pad>": 0, "<unk>": 1
}

for i, syll in enumerate(sorted(syllable_counter), start=2):
    vocab[syll] = i

with open("data/syllabert_phone/phone_vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)

# with open("data/syllabert_phone/phone_vocab.json", 'r') as f:
#     vocab = json.load(f)


eval_ratio = 0.1
eval_num = int(len(audio_paths) * eval_ratio)


with open("data/syllabert_phone/phone_manifest_val.jsonl", 'w') as f:
    for audio_path in tqdm(audio_paths[:eval_num]):
        utterance_id = os.path.splitext(os.path.basename(audio_path))[0]
        tg_path = os.path.join(
                    textgrid_dir,
                    os.path.relpath(audio_path, audio_dir).replace(".flac", "_syllabified.TextGrid")
                )
        if not os.path.exists(textgrid_path):
            print(f"Skippng {audio_path} as the corresponding textgrid does not exist.")
            continue

        tg = TextGrid.fromFile(tg_path)
        entry = {
            "file": audio_path,
            "uttr_id": utterance_id
        }
        info = []
        for tier in tg.tiers:
            if "phones" in tier.name.lower():
                # syllable_tier = tier
                # break
                for interval in tier:
                    m = interval.mark.strip()
                    if m:
                        # entry = {
                        #     "file": audio_path,
                        #     "uttr_id": utterance_id,
                        #     "segment_id": idx,
                        #     "segment_start": float(interval.minTime),
                        #     "segment_end": float(interval.maxTime),
                        #     "phone_idx": int(vocab[interval.mark])
                        # }
                        info.append([
                            float(interval.minTime),
                            float(interval.maxTime),
                            int(vocab[interval.mark])
                        ])
        entry['info'] = info
        json.dump(entry, f)
        f.write('\n')


with open("data/syllabert_phone/phone_manifest_train.jsonl", 'w') as f:
    for audio_path in tqdm(audio_paths[eval_num:]):
        utterance_id = os.path.splitext(os.path.basename(audio_path))[0]
        tg_path = os.path.join(
                    textgrid_dir,
                    os.path.relpath(audio_path, audio_dir).replace(".flac", "_syllabified.TextGrid")
                )
        if not os.path.exists(textgrid_path):
            print(f"Skippng {audio_path} as the corresponding textgrid does not exist.")
            continue

        tg = TextGrid.fromFile(tg_path)
        entry = {
            "file": audio_path,
            "uttr_id": utterance_id
        }
        info = []
        for tier in tg.tiers:
            if "phones" in tier.name.lower():
                # syllable_tier = tier
                # break
                for interval in tier:
                    m = interval.mark.strip()
                    if m:
                        # entry = {
                        #     "file": audio_path,
                        #     "uttr_id": utterance_id,
                        #     "segment_id": idx,
                        #     "segment_start": float(interval.minTime),
                        #     "segment_end": float(interval.maxTime),
                        #     "phone_idx": int(vocab[interval.mark])
                        # }
                        info.append([
                            float(interval.minTime),
                            float(interval.maxTime),
                            int(vocab[interval.mark])
                        ])
        entry['info'] = info
        json.dump(entry, f)
        f.write('\n')
