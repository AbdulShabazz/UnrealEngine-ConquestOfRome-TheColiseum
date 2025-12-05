import os
import csv
from pathlib import Path

import torch
import torchaudio
from transformers import AutoProcessor, ClapModel

# ---------- CONFIG ----------

AUDIO_DIR = r"D:\BoomAudio"
OUTPUT_CSV = r"D:\sound_tags.csv"

TOP_K = 3  # number of labels per sound

MODEL_ID = "laion/clap-htsat-fused"  # zero-shot audio-text model

GLADIATOR_LABELS = {
    "melee_sword_slash_light":
        "short metal sword slash, close combat, gladius cutting air or light armor, dry, close mic",
    "melee_sword_slash_heavy":
        "heavy metal sword swing with strong impact, brutal gladiator strike, lots of low mid energy",
    "melee_spear_thrust":
        "fast spear or javelin thrust, sharp attack transient, point piercing target",
    "melee_mace_blunt":
        "blunt metal mace hit on armor, heavy low thump with metallic clank",
    "hit_armor_metal_light":
        "light impact on metal armor, small shield tap, subtle metallic ring",
    "hit_armor_metal_heavy":
        "strong metal impact on armor or shield, loud clang and ring, gladiator blocking attack",
    "hit_flesh_light":
        "light flesh hit, muted wet thud, little or no armor",
    "hit_flesh_heavy":
        "heavy body impact with flesh and bone, thick low frequency thump, possible gore",
    "shield_block_wood":
        "wooden shield block, woody thud with small rattle, roman scutum",
    "shield_block_metal":
        "metal shield block, plate clash, ringing edge",
    "footstep_sand":
        "human footsteps on sand or dirt arena floor, soft granular scrapes",
    "footstep_stone":
        "sandaled footsteps on stone, hard clicks and small reverbs, coliseum corridors",
    "vocal_battle_shout":
        "single male gladiator battle shout or yell, aggressive voice, medium reverb",
    "vocal_pain_cry":
        "human pain scream or grunt, hurt vocalization in combat",
    "crowd_cheer_large":
        "large ancient roman coliseum crowd cheering, wide stereo ambience, wall of voices",
    "crowd_boo":
        "crowd booing or disapproving, descending vocal reactions",
    "crowd_murmur_idle":
        "large crowd idle murmur, distant talking ambience, no big cheer",
    "environment_fire":
        "fire or torches burning, crackling flames, medieval ambience",
    "environment_chain_rattle":
        "chains rattling, metal links clanking, dungeon like",
    "ui_menu":
        "short clean user interface click or confirm sound, non diegetic, no ambience",
    "ui_error":
        "short clean user interface error or cancel sound, non diegetic",
}

# ---------- MODEL LOAD ----------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading CLAP model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = ClapModel.from_pretrained(MODEL_ID).to(device)
model.eval()

label_names = list(GLADIATOR_LABELS.keys())
label_texts = [GLADIATOR_LABELS[k] for k in label_names]

# Pre-encode label texts once
with torch.no_grad():
    text_inputs = processor(text=label_texts, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    text_features = model.get_text_features(**text_inputs)  # [num_labels, dim]
    text_features = torch.nn.functional.normalize(text_features, dim=-1)


def classify_audio(path: Path, top_k: int = 3):
    waveform, sr = torchaudio.load(str(path))

    # CLAP expects 48 kHz by default; resample if needed.:contentReference[oaicite:3]{index=3}
    target_sr = 48000
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    # Convert to mono if multi-channel
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    audio_array = waveform.squeeze(0).numpy()

    with torch.no_grad():
        inputs = processor(audios=audio_array, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        audio_features = model.get_audio_features(**inputs)  # [1, dim]
        audio_features = torch.nn.functional.normalize(audio_features, dim=-1)

        # cosine similarity ~ dot product after normalization
        sims = (audio_features @ text_features.T).squeeze(0)  # [num_labels]

        topk_vals, topk_idx = torch.topk(sims, k=top_k)
        results = []
        for score, idx in zip(topk_vals.tolist(), topk_idx.tolist()):
            results.append((label_names[idx], float(score)))
        return results


def iter_wavs(root: Path):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".wav", ".flac", ".ogg")):
                yield Path(dirpath) / fn

def main():
    pass
"""
    root = Path(AUDIO_DIR)
    rows = []

    for audio_path in iter_wavs(root):
        rel = audio_path.relative_to(root)
        print(f"Processing {rel}...")
        try:
            tags = classify_audio(audio_path, top_k=TOP_K)
        except Exception as e:
            print(f"  ERROR on {rel}: {e}")
            continue

        row = {
            "relative_path": str(rel),
        }
        for i, (label, score) in enumerate(tags, start=1):
            row[f"tag_{i}"] = label
            row[f"score_{i}"] = f"{score:.4f}"

        rows.append(row)

    fieldnames = ["relative_path"]
    for i in range(1, TOP_K + 1):
        fieldnames += [f"tag_{i}", f"score_{i}"]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")
"""

if __name__ == "__main__":
    main()
