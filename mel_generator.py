import os
import tempfile
from gtts import gTTS
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# ------------------------------------------------------------
# Rutas
# ------------------------------------------------------------
os.makedirs("audios", exist_ok=True)
os.makedirs("mels", exist_ok=True)
# Lista de vocales a sintetizar
vocales = ['a', 'e', 'i', 'o', 'u']

def generar_audio_y_mel(vocal):
    """
        Genera un audio con gTTS para la vocal dada y su espectrograma Mel.
    """

    # Texto simple repetido para alargar la vocal
    tts_text = vocal * 12
    tts = gTTS(text=tts_text, lang='es', slow=True)  # 'slow=True' alarga y clarifica

    # Guardar a un MP3 temporal
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_mp3 = tmp.name
        tts.save(tmp_mp3)

    # Convertir el MP3 temporal a WAV definitivo y eliminar el MP3
    y, sr = librosa.load(tmp_mp3, sr=None)
    sf.write(f"audios/{vocal}.wav", y, sr)
    try:
        os.remove(tmp_mp3)
    except OSError:
        pass

    # Espectrograma Mel
    S = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=1024,
        hop_length=128,
        n_mels=256,
        fmax=8000
    )
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Guardar imagen
    plt.figure(figsize=(8, 4.5), dpi=200)
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(img, format='%+2.0f dB')
    plt.title(f'Espectrograma Mel - Vocal "{vocal}"')
    plt.tight_layout()
    plt.savefig(f"mels/{vocal}_mel.png", bbox_inches='tight')
    plt.close()

    return S_dB

# ------------------------------------------------------------
# Ejecutar para todas las vocales y guardar resultados
# ------------------------------------------------------------
mels_dict = {v: generar_audio_y_mel(v) for v in vocales}

print("✅ WAVs en 'audios/' y espectrogramas mel en 'mels/'. Sin conservar MP3.")
