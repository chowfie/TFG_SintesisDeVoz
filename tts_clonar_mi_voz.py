import torch
from TTS.api import TTS

# Detectar si hay GPU disponible
device = "cuda" if torch.cuda.is_available() else "cpu"

# Texto a sintetizar
txt = "Bienvenido a este nuevo artículo del blog. Disfruta de tu visita."

# Ruta al audio de referencia
sample = "./audios_TFG/wavs/audio49.wav"

# Cargar modelo XTTS y moverlo al dispositivo
tts1 = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Generar archivo de audio con voz clonada
tts1.tts_to_file(txt, speaker_wav=sample, language="es", file_path="./pruebas/mi_voz1.wav")
