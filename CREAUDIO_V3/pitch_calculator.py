#   python .\pitch_per_speaker.py

from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# -----------------------------
# Rutas
# -----------------------------
CV_UTT   = Path(r"C:\Users\sreal\Desktop\TFG\CODIGO\CREAUDIO_V3\embeddings_cv\embeddings_cv_per_utterance.csv")
VCTK_UTT = Path(r"C:\Users\sreal\Desktop\TFG\CODIGO\CREAUDIO_V3\embeddings_vctk\embeddings_vctk_per_utterance.csv")
LIB_UTT  = Path(r"C:\Users\sreal\Desktop\TFG\CODIGO\CREAUDIO_V3\embeddings_librittsr\embeddings_libritts_per_utterance.csv")

OUT_CSV = Path("./latent_out/pitch_per_speaker.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Pitch de un audio
# -----------------------------
def f0_median_for_file(wav_path: Path, sr: int = 22050) -> float:
    """
        Carga un audio y saca su F0 (frecuencia fundamental).
        Devuelve la mediana de F0 en Hz, o NaN si no se puede.
    """
    try:
        y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
        # recorte sencillo de silencios
        y, _ = librosa.effects.trim(y, top_db=30)
        if y.size < 0.3 * sr:  # si queda < 0.3s, pasa
            return np.nan

        f0, _, _ = librosa.pyin(
            y, fmin=50, fmax=450, sr=sr,
            frame_length=int(0.04 * sr), hop_length=int(0.01 * sr),
        )
        f0 = f0[~np.isnan(f0)]
        if len(f0) == 0:
            return np.nan
        return float(np.median(f0))
    except Exception:
        return np.nan

# -----------------------------
# Pitch per speaker
# -----------------------------
def process_csv(path: Path, corpus: str, max_utts: int = 20) -> pd.DataFrame:
    """
        Lee el CSV por utterance de un corpus, calcula pitch de cada audio,
        y devuelve un DataFrame con el pitch medio por speaker.
    """
    if not path.exists():
        print(f"[WARN] No encuentro {path}, lo salto.")
        return pd.DataFrame(columns=["corpus","speaker_id","PITCH_HZ","num_utts_used"])

    df = pd.read_csv(path)
    if not {"speaker_id","audio_path"}.issubset(df.columns):
        raise ValueError(f"{path} no tiene columnas speaker_id/audio_path")

    rows = []
    for spk, g in tqdm(df.groupby("speaker_id"), desc=f"{corpus} speakers"):
        utts = g["audio_path"].tolist()[:max_utts]  # límite de audios por speaker
        f0_vals = [f0_median_for_file(Path(u)) for u in utts]
        f0_vals = [v for v in f0_vals if not np.isnan(v)]
        if len(f0_vals) == 0:
            pitch = np.nan
        else:
            pitch = float(np.median(f0_vals))
        rows.append({
            "corpus": corpus,
            "speaker_id": spk,
            "PITCH_HZ": pitch,
            "num_utts_used": len(f0_vals)
        })
    return pd.DataFrame(rows)

# -----------------------------
# Main
# -----------------------------
all_dfs = []
all_dfs.append(process_csv(CV_UTT,   "CommonVoice-ES"))
all_dfs.append(process_csv(VCTK_UTT, "VCTK"))
all_dfs.append(process_csv(LIB_UTT,  "LibriTTS-R"))

final = pd.concat(all_dfs, ignore_index=True)
final["LOG_F0"] = np.log(final["PITCH_HZ"].clip(lower=1.0))
final.to_csv(OUT_CSV, index=False)

print(f"[OK] Guardado {OUT_CSV.resolve()}")
print(final.head())