import os
import soundfile as sf
from TTS.utils.synthesizer import Synthesizer
import torch
from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np

# -------------------------------
# CONFIGURACIÓN
# -------------------------------
USE_CUDA = torch.cuda.is_available()

SOURCE_AUDIO = "C:/Users/sreal/Desktop/TFG/CODIGO/LJSPEECH/LJ001-0001.wav"
REFERENCE_AUDIO = "C:/Users/sreal/Desktop/TFG/AUDIOS MI VOZ/wavs_16Khz/audio41.wav"
OUTPUT_AUDIO = "output_results/converted.wav"

os.makedirs(os.path.dirname(OUTPUT_AUDIO), exist_ok=True)

# Ruta del modelo descargado (por ejemplo con la API de Coqui)
MODEL_DIR = "C:/Users/sreal/Desktop/TFG/CODIGO/models"  # actualiza si lo moviste

# Archivos necesarios
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "model.pth")
SPEAKER_ENCODER_PATH = os.path.join(MODEL_DIR, "speaker_encoder.pth")
SPEAKER_ENCODER_CONFIG = os.path.join(MODEL_DIR, "speaker_encoder_config.json")
VOCODER_PATH = CHECKPOINT_PATH  # XTTS ya incluye vocoder

# -------------------------------
# CARGA DE SYNTHESIZER
# -------------------------------
print("🧠 Cargando Synthesizer...")
synth = Synthesizer(
    tts_checkpoint=CHECKPOINT_PATH,
    tts_config_path=CONFIG_PATH,
    encoder_checkpoint=SPEAKER_ENCODER_PATH,
    encoder_config=SPEAKER_ENCODER_CONFIG,
    vocoder_checkpoint=VOCODER_PATH,
    vocoder_config=CONFIG_PATH,
    use_cuda=USE_CUDA,
)

# -------------------------------
# TRANSFERENCIA DE VOZ
# -------------------------------
print("🎙️ Transfiriendo voz (voice conversion)...")
converted = synth.tts(reference_wav=REFERENCE_AUDIO, audio_path=SOURCE_AUDIO)

sf.write(OUTPUT_AUDIO, converted, samplerate=22050)
print(f"✅ Voz convertida guardada en: {OUTPUT_AUDIO}")

# -------------------------------
# SIMILITUD CON RESEMBLYZER
# -------------------------------
print("📊 Analizando similitud acústica...")

encoder = VoiceEncoder()

def get_embedding(path):
    wav = preprocess_wav(path)
    return encoder.embed_utterance(wav)

emb_source = get_embedding(SOURCE_AUDIO)
emb_reference = get_embedding(REFERENCE_AUDIO)
emb_converted = get_embedding(OUTPUT_AUDIO)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sim_content = cosine_sim(emb_source, emb_converted)
sim_voice = cosine_sim(emb_reference, emb_converted)

print(f"\n🔍 Resultados de similitud (coseno):")
print(f"- Contenido (original vs convertido): {sim_content:.4f}")
print(f"- Voz (referencia vs convertido):     {sim_voice:.4f}")
