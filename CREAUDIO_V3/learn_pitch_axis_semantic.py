# python learn_pitch_axis_semantic.py

from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Rutas y ajustes
# ------------------------------------------------------------
LATENT_DIR = Path("latent_out")
EMB_CSV    = LATENT_DIR / "embeddings_with_metadata.csv"  # generado por learn_axes.py
PITCH_CSV  = LATENT_DIR / "pitch_per_speaker.csv"         # tabla por speaker con LOG_F0
OUT_NPZ    = LATENT_DIR / "semantic_directions.npz"       # contenedor de direcciones 512D

# Columnas y percentiles
SPEAKER_COL = "speaker_id"
CORPUS_COL  = "corpus"
PITCH_COL   = "LOG_F0"   # usar LOG_F0 suele ser más robusto perceptualmente
LOW_Q       = 0.20       # percentil bajo para agrupar voces graves
HIGH_Q      = 0.80       # percentil alto para agrupar voces agudas

# ------------------------------------------------------------
# Funciones auxiliares
# ------------------------------------------------------------
def unit(v, eps=1e-9):
    """
    Normaliza el vector 'v' a norma L2 = 1.
    'eps' evita divisiones por cero si el vector es casi nulo.
    """
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return (v / max(n, eps)).astype(np.float32)

# ------------------------------------------------------------
# Función principal
# ------------------------------------------------------------
def main():

    # Cargar embeddings con metadatos
    if not EMB_CSV.exists():
        raise FileNotFoundError(f"No encuentro {EMB_CSV}. Ejecuta learn_axes.py primero.")
    df = pd.read_csv(EMB_CSV, low_memory=False)

    # Detectar columnas de embedding e0...eN (512D)
    emb_cols = [c for c in df.columns if c.startswith("e")]
    if not emb_cols:
        raise RuntimeError("El CSV no tiene columnas e0..eN (embeddings 512D).")

    # Cargar pitch por speaker
    if not PITCH_CSV.exists():
        raise FileNotFoundError(f"No encuentro {PITCH_CSV}. Coloca 'pitch_per_speaker.csv' en latent_out.")
    p = pd.read_csv(PITCH_CSV)

    # Unir por corpus + speaker_id cuando sea posible, si no, por speaker_id
    on_cols = []
    if CORPUS_COL in df.columns and CORPUS_COL in p.columns:
        on_cols = [CORPUS_COL, SPEAKER_COL]
    else:
        on_cols = [SPEAKER_COL]

    m = pd.merge(df, p, on=on_cols, how="inner")
    if m.empty:
        raise RuntimeError("La unión embeddings+pitch quedó vacía. Revisa las claves y los CSV.")

    if PITCH_COL not in m.columns:
        raise RuntimeError(f"No existe la columna '{PITCH_COL}' en {PITCH_CSV}. Columnas: {list(p.columns)}")

    # Seleccionar extremos por LOG_F0 usando percentiles
    x = pd.to_numeric(m[PITCH_COL], errors="coerce")
    lo = np.nanpercentile(x, LOW_Q * 100)
    hi = np.nanpercentile(x, HIGH_Q * 100)

    mask_low  = x <= lo   # grupo de voces graves
    mask_high = x >= hi   # grupo de voces agudas

    n_low, n_high = int(mask_low.sum()), int(mask_high.sum())
    if n_low < 8 or n_high < 8:
        print(f"Aviso: pocos speakers en extremos ({n_low} low, {n_high} high). Ajusta LOW_Q/HIGH_Q si hace falta.")

    # Matriz de embeddings 512D y medias por grupo
    X = m[emb_cols].to_numpy(dtype=np.float32)
    mu_low  = X[mask_low].mean(axis=0)
    mu_high = X[mask_high].mean(axis=0)

    # Dirección 512D: alto pitch menos bajo pitch
    # Positivo moverá hacia más agudo, negativo hacia más grave.
    pitch_high_minus_low = unit(mu_high - mu_low)

    # Guardar en semantic_directions.npz sin borrar lo anterior
    existing = {}
    if OUT_NPZ.exists():
        existing = dict(np.load(OUT_NPZ, allow_pickle=True))
    existing["pitch_high_minus_low"] = pitch_high_minus_low
    np.savez(OUT_NPZ, **existing)

# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
