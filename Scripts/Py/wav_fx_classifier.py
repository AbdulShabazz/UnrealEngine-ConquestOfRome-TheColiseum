import os
#os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "1"

#import csv
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
from transformers import AutoProcessor, ClapModel
from tqdm import tqdm
from collections import defaultdict

# ---------- CONFIG ----------

AUDIO_DIR = r"D:\BoomAudio"
INPUT_TXT = r"D:\crc_undex.txt"
OUTPUT_TXT = r"D:\crc_audio_rankings.txt"

TOP_K = 3  # number of labels per sound

MODEL_ID = "laion/clap-htsat-fused"  # zero-shot audio-text model

def load_labels_from_crc(filepath: str = "crc_index.txt") -> dict[str, str]:
    labels = {}
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue  # skip headers/empty lines
            
            # Split taxonomy path from description
            taxonomy_part, description = line.split(":", 1)
            
            # Extract last segment as key: "soundFxTaxonomy__Ambient Environment__Coliseum Ambient"
            segments = taxonomy_part.split("__")
            key = segments[-1].strip().lower().replace(" ", "_").replace("/", "_")
            
            labels[key] = description.strip()
    
    return labels

GLADIATOR_LABELS = load_labels_from_crc("crc_index.txt") or {
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

# torchaudio.set_audio_backend("soundfile")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading CLAP model...")

target_sr = 48000   # CLAP expects 48kHz

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
    # waveform, sr = torchaudio.load(str(path))

    audio_array, sr = sf.read(str(path))  # returns numpy array
    waveform = torch.from_numpy(audio_array.T).float()  # [channels, samples]

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # CLAP expects 48 kHz by default; resample if needed.:contentReference[oaicite:3]{index=3}
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    # Minimum length check in classify_audio
    MIN_SAMPLES = target_sr  # 1 second minimum
    if waveform.shape[-1] < MIN_SAMPLES:
        waveform = torch.nn.functional.pad(waveform, (0, MIN_SAMPLES - waveform.shape[-1]))

    # Convert to mono if multi-channel
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    audio_array = waveform.squeeze(0).numpy()

    with torch.no_grad():
        inputs = processor(audio=audio_array, sampling_rate=sr, return_tensors="pt", padding=True)
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


def iter_wavs(root: Path, recursive: bool = False):
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith((".wav", ".flac", ".ogg")):
                    yield Path(dirpath) / fn
    else:
        for fn in root.iterdir():
            if fn.is_file() and fn.suffix.lower() in (".wav", ".flac", ".ogg"):
                yield fn

def main():
    
    root = Path(AUDIO_DIR)

    # Potential empty directory handling:
    if not root.exists():
        raise FileNotFoundError(f"Audio directory not found: {root}")

    rows = []

    # Progress feedback for large libraries:
    pbar = tqdm(list(iter_wavs(root, recursive=True)), desc="Classifying", position=0, leave=True)  # patch updates screen stacking
    for audio_path in pbar:
        rel = audio_path.relative_to(root)
        pbar.set_description(f"{str(rel):<250}")     
        try:
            tags = classify_audio(audio_path, top_k=TOP_K)
        except Exception as e:
            tqdm.write(f"  ERROR on {rel}: {e}")    # tqdm-safe output
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

    # Group files by top tag
    categorized = defaultdict(list)
    for row in rows:
        top_tag = row["tag_1"]
        categorized[top_tag].append(row["relative_path"])

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for category in sorted(categorized.keys()):
            f.write(f"[{category}]\n")
            for path in sorted(categorized[category]):
                f.write(f"  {path}\n")
            f.write("\n")

    print(f"Wrote {len(rows)} rows to {OUTPUT_TXT}")   

if __name__ == "__main__":
    main()
