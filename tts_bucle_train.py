# python tts_bucle_train.py

import torch
from TTS.api import TTS
from TTS.tts.utils.managers import EmbeddingManager
from resemblyzer import VoiceEncoder, preprocess_wav
import os
import shutil
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import soundfile as sf
from pathlib import Path


# -------------------------------
# UTILS
# -------------------------------
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def convert_float32_to_float(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_float32_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_float32_to_float(v) for v in obj]
    return obj

def get_audio_duration(file_path):
    with sf.SoundFile(file_path) as f:
        return len(f) / f.samplerate


def convert_flac_to_wav(flac_path, out_dir):
    wav_path = os.path.join(out_dir, Path(flac_path).stem + ".wav")
    if not os.path.exists(wav_path):
        audio, sr = sf.read(flac_path)
        sf.write(wav_path, audio, samplerate=sr)
    return wav_path

# -------------------------------
# CONFIGURATION
# -------------------------------
# Set Paths
USE_CUDA = torch.cuda.is_available()
OUTPUT_BASE_DIR = "output_results_vctk2"
# Local paths to encoder model
ENCODER_MODEL_PATH = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar"
ENCODER_CONFIG_PATH = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json"

MAX_TRAIN_AUDIOS = 250
TEST_STEP = 25

# Create output directory
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE_DIR, "cloned_audios"), exist_ok=True)

# -------------------------------
# INITIALIZATION
# -------------------------------
print("Initializing embedding manager...")
embedding_manager = EmbeddingManager(
    encoder_model_path=ENCODER_MODEL_PATH,
    encoder_config_path=ENCODER_CONFIG_PATH,
    use_cuda=USE_CUDA
)

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
print(f"Using TTS model: {MODEL_NAME}")
synthesizer = TTS(model_name=MODEL_NAME, progress_bar=True)
synthesizer.to("cuda" if USE_CUDA else "cpu")

# -------------------------------
# LOAD SPEAKER AUDIOS
# -------------------------------
print("\nLoading audios...")
AUDIOS_BASE_PATH = r"C:\Users\sreal\Desktop\TFG\CODIGO\datasets\VCTK-Corpus-0.92\wav48_silence_trimmed"
# For VCTK audios, select a speaker
SELECTED_SPEAKER = "p256"
print(f"\nSearching in folder: {AUDIOS_BASE_PATH}\\{SELECTED_SPEAKER}")

speaker_folder = os.path.join(AUDIOS_BASE_PATH, SELECTED_SPEAKER)

# Search FLAC and WAV files
flac_files = sorted(glob(os.path.join(speaker_folder, "*.flac")))
wav_files = sorted(glob(os.path.join(speaker_folder, "*.wav")))

# If WAVs found, use them directly
if wav_files:
    print("→ Using WAV files.")
    all_speaker_audios = wav_files
elif flac_files:
    print("→ Found FLAC files. Doing temporal conversion to WAV.")
    all_speaker_audios = flac_files
else:
    raise ValueError(f"WAV or FLAC files not found in this folder: {speaker_folder}")


# Divide 90% audios for training and 10% for testing
split_index = int(0.9 * len(all_speaker_audios))
train_audios = all_speaker_audios[:split_index][:MAX_TRAIN_AUDIOS]
test_audios = all_speaker_audios[split_index:]

# -------------------------------
# FLAC → WAV CONVERSION (IF NEEDED)
# -------------------------------
if flac_files:
    # Save wav files in temporary folder
    TEMP_WAV_DIR = os.path.join("temp_wavs", SELECTED_SPEAKER)
    os.makedirs(TEMP_WAV_DIR, exist_ok=True)

    # Convert audios
    train_audios = [convert_flac_to_wav(f, TEMP_WAV_DIR) for f in train_audios]
    test_audios = [convert_flac_to_wav(f, TEMP_WAV_DIR) for f in test_audios]

assert all(os.path.getsize(f) > 0 for f in train_audios + test_audios), "Some audio files are empty or corrupted."
print(f"→ Locutor: {SELECTED_SPEAKER}")
print(f"→ Training audios: {len(train_audios)}")
print(f"→ Test audios: {len(test_audios)}")

# -------------------------------
# PRECOMPUTE TEST EMBEDDINGS
# -------------------------------
print("\nPrecomputing test embeddings...")
test_embeddings = [
    embedding_manager.compute_embedding_from_clip(a) for a in tqdm(test_audios, desc="Test embeddings")
]

# -------------------------------
# STEPS TO EVALUATE
# -------------------------------
steps = list(range(1, 11)) + list(range(25, MAX_TRAIN_AUDIOS + 1, TEST_STEP))
steps = sorted(list(set(steps)))  # Ensure increasing order without duplicates

results = {
    'duration_reference_audios_sec': [],
    'avg_similarity': [],
    'std_similarity': [],
    'all_similarities': []
}

texts = [
    "This is an example text for voice cloning.",
    "She sells seashells by the seashore.",
    "The quick brown fox jumps over the lazy dog.",
    "Can you hear the difference in my voice?",
    "I love machine learning and synthetic voices."
]

encoder = VoiceEncoder()
resemblyzer_results_multi = {
    "duration_reference_audios_sec": [],
    "avg_similarity": [],
    "std_similarity": [],
}

print("\nEvaluating steps...")
for num_ref in tqdm(steps, desc="Evaluating steps"):
    # Audios to use in this iteration
    current_ref = train_audios[:num_ref]
    # Calculate embedding
    combined_embedding = embedding_manager.compute_embedding_from_clip(current_ref)

    # Create folder for results in this iteration
    iteration_dir = os.path.join(OUTPUT_BASE_DIR, "cloned_audios", f"iter_{num_ref}")
    os.makedirs(iteration_dir, exist_ok=True)

    similarities_all = []
    for i, sentence in enumerate(texts):
        # Synthesize
        wav = synthesizer.tts(
            text=sentence,
            speaker_wav=current_ref,
            language="en"
        )

        # Save audio for sentence
        output_path = os.path.join(iteration_dir, f"sentence_{i + 1}.wav")
        sf.write(output_path, wav, samplerate=22050)

        # Embedding con Resemblyzer
        try:
            ref_wav = preprocess_wav(output_path)
            ref_emb = encoder.embed_utterance(ref_wav)

            for test_path in test_audios:
                test_wav = preprocess_wav(test_path)
                test_emb = encoder.embed_utterance(test_wav)
                sim = cosine_similarity(ref_emb, test_emb)
                similarities_all.append(sim)
        except Exception as e:
            print(f"Error in Resemblyzer processing: {e}")
            continue

    if similarities_all:
        total_duration = sum(get_audio_duration(f) for f in current_ref)
        resemblyzer_results_multi["duration_reference_audios_sec"].append(total_duration)
        resemblyzer_results_multi["avg_similarity"].append(np.mean(similarities_all))
        resemblyzer_results_multi["std_similarity"].append(np.std(similarities_all))

    # Calculate similarities with test embeddings
    test_similarities = [
        cosine_similarity(combined_embedding, test_emb)
        for test_emb in test_embeddings
    ]

    # Store results
    total_duration = sum(get_audio_duration(f) for f in current_ref)
    results['duration_reference_audios_sec'].append(total_duration)

    results['avg_similarity'].append(np.mean(test_similarities))
    results['std_similarity'].append(np.std(test_similarities))
    results['all_similarities'].append(test_similarities)

# -------------------------------
# Save results as JSON
# -------------------------------
with open(os.path.join(OUTPUT_BASE_DIR, "embedding_results.json"), "w") as f:
    json.dump(results, f, indent=4)

with open(os.path.join(OUTPUT_BASE_DIR, "resemblyzer_results_multi.json"), "w") as f:
    json.dump(convert_float32_to_float(resemblyzer_results_multi), f, indent=4)

# -------------------------------
# Plot similarity evolution
# -------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.errorbar(
    results['duration_reference_audios_sec'],
    results['avg_similarity'],
    yerr=results['std_similarity'],
    fmt='o-',
    capsize=5,
    elinewidth=1.5,
    marker='o',
    markersize=6,
    color='#2ca02c',
    ecolor='lightgreen',
    alpha=0.9,
    label='Average similarity ± std. deviation'
)
ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.4, label='Maximum possible similarity')
plt.title("Embedding similarity vs reference audio duration", fontsize=15)
plt.xlabel("Total duration of reference audios (s)", fontsize=12)
plt.ylabel("Cosine similarity", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
ax.legend(fontsize=10)
plt.tight_layout()
ax.grid(True, linestyle='--', alpha=0.3)
plt.savefig(os.path.join(OUTPUT_BASE_DIR, "embedding_similarity_evolution.png"), dpi=300)
plt.show()

# -------------------------------
# PLOT RESEMBLYZER SIMILARITY
# -------------------------------
'''
Resemblyzer proporciona una métrica perceptiva útil para estimar la similitud de la voz, pero no siempre refleja 
de forma lineal la mejora de calidad con más audios de referencia. A partir de cierto punto, los embeddings tienden 
a generalizar la identidad vocal, lo que puede reducir la similitud exacta con audios originales sin que ello implique 
una pérdida real de naturalidad o claridad en la voz clonada.
'''

plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(12, 6))
plt.errorbar(
    resemblyzer_results_multi["duration_reference_audios_sec"],
    resemblyzer_results_multi["avg_similarity"],
    yerr=resemblyzer_results_multi["std_similarity"],
    fmt='o-',
    capsize=5,
    elinewidth=1.5,
    marker='o',
    markersize=6,
    color='#1f77b4',
    ecolor='lightblue',
    alpha=0.9,
    label="Resemblyzer similarity ± std."
)
plt.axhline(y=0.83, color='gray', linestyle='--', alpha=0.5, label="Reference mean")
ax.set_title("Voice similarity with multiple test sentences", fontsize=15, weight='bold')
ax.set_xlabel("Total duration of reference audios (seconds)", fontsize=12)
ax.set_ylabel("Resemblyzer similarity score", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_BASE_DIR, "resemblyzer_similarity.png"), dpi=300)
plt.show()

# -------------------------------
# FINAL ANALYSIS
# -------------------------------
final_avg = results['avg_similarity'][-1]
initial_avg = results['avg_similarity'][0]
improvement = (final_avg - initial_avg) / initial_avg * 100

print("\nFinal analysis:")
print(f"- Initial similarity (1 audio): {initial_avg:.4f}")
print(f"- Final similarity ({MAX_TRAIN_AUDIOS} audios): {final_avg:.4f}")
print(f"- Relative improvement: {improvement:.2f}%")

# -------------------------------
# Delete temporary folder
# -------------------------------

if flac_files:
    temp_dir = "temp_wavs"

    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print(f"🧹 Temporary folder deleted: {temp_dir}")
        except Exception as e:
            print(f"❌ Error deleting temporary folder '{temp_dir}': {e}")