# python learn_pitch_axis_Z.py

from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load as joblib_load

# ------------------------------------------------------------
# Rutas y ajustes
# ------------------------------------------------------------
LATENT_DIR = Path("./latent_out")
EMB_CSV    = LATENT_DIR / "embeddings_with_metadata.csv"
PITCH_CSV  = LATENT_DIR / "pitch_per_speaker.csv"
OUT_NPZ    = LATENT_DIR / "semantic_z_from_corr.npz"
OUT_CSV    = LATENT_DIR / "correlaciones_Z_con_pitch.csv"

PCA_DIR       = Path("./pca_xtts_k64")
SCALER_JOBLIB = PCA_DIR / "scaler.joblib"
PCA_JOBLIB    = PCA_DIR / "pca.joblib"

SPEAKER_COL = "speaker_id"
CORPUS_COL  = "corpus"
PITCH_COL   = "LOG_F0"     # usar LOG_F0 es mas robusto perceptualmente
TOPK_FOR_Z_DIRS = 3        # numero de z_i que combinamos por |r|

# ------------------------------------------------------------
# Funciones auxiliares
# ------------------------------------------------------------
def unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Normaliza un vector a norma L2 = 1. Evita division por cero con eps.
    """
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return (v / max(n, eps)).astype(np.float32)

def build_corr_based_direction_in_Z(corr_df: pd.DataFrame, k_total: int, topk: int) -> np.ndarray:
    """
        Construye una direccion en Z combinando las top-k componentes por |r|,
        ponderadas por |r| y conservando el signo de la correlacion.
        - k_total: numero total de componentes Z
        - topk: numero de z_i a usar segun mayor |r|
    """
    use = corr_df.head(max(1, topk)).copy()

    # Pesos proporcionales a |r|. Si todo es ~0, usar 1.0 para evitar division por cero.
    use["w"] = use["pearson_r"].abs()
    if use["w"].sum() <= 1e-9:
        use["w"] = 1.0
    use["w"] = use["w"] / use["w"].sum()

    # Vector direccion en Z con longitud k_total
    v = np.zeros((k_total,), dtype=np.float32)
    for _, row in use.iterrows():
        comp = str(row["component"])   # por ejemplo, "z3"
        r    = float(row["pearson_r"])
        idx  = int(comp.replace("z", "")) - 1  # "z3" -> indice 2 (0-based)
        sign = 1.0 if r >= 0 else -1.0
        v[idx] += sign * float(row["w"])       # acumula peso con signo

    return unit(v)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    # Cargar embeddings con metadatos
    if not EMB_CSV.exists():
        raise FileNotFoundError(f"No encuentro {EMB_CSV}. Ejecuta learn_axes.py primero.")
    df = pd.read_csv(EMB_CSV, low_memory=False)

    # Columnas e0...eN
    emb_cols = [c for c in df.columns if c.startswith("e")]
    if not emb_cols:
        raise RuntimeError("El CSV no tiene columnas e0..eN (embeddings 512D).")

    # Cargar pitch por speaker
    if not PITCH_CSV.exists():
        raise FileNotFoundError(f"No encuentro {PITCH_CSV}. Debe existir en {LATENT_DIR}.")
    p = pd.read_csv(PITCH_CSV)

    # Union por claves (mejor corpus+speaker si existen, si no solo speaker)
    on_cols = [CORPUS_COL, SPEAKER_COL] if (CORPUS_COL in df.columns and CORPUS_COL in p.columns) else [SPEAKER_COL]
    m = pd.merge(df, p, on=on_cols, how="inner")
    if m.empty:
        raise RuntimeError("La union embeddings_with_metadata.csv + pitch_per_speaker.csv quedo vacia.")
    if PITCH_COL not in m.columns:
        raise RuntimeError(f"No existe la columna '{PITCH_COL}' en {PITCH_CSV}. Columnas: {list(p.columns)}")

    # Cargar PCA y scaler
    if not (SCALER_JOBLIB.exists() and PCA_JOBLIB.exists()):
        raise FileNotFoundError("Faltan scaler.joblib o pca.joblib. Entrena PCA con pca_train.py.")
    scaler = joblib_load(SCALER_JOBLIB)
    pca    = joblib_load(PCA_JOBLIB)

    # Proyeccion a Z para las filas unidas
    X  = m[emb_cols].to_numpy(dtype=np.float32)
    Xs = scaler.transform(X)
    Z  = pca.transform(Xs)  # shape: (n, k)
    k  = Z.shape[1]
    zcols = [f"z{i+1}" for i in range(k)]

    # Correlaciones z_i vs LOG_F0 (Pearson)
    zdf = pd.DataFrame(Z, columns=zcols)
    y   = pd.to_numeric(m[PITCH_COL], errors="coerce")

    rows = []
    for c in zcols:
        # corr() devuelve matriz 2x2 y tomamos [0,1] como r
        r = pd.concat([zdf[c], y], axis=1).corr(method="pearson").iloc[0, 1]
        rows.append({"component": c, "pearson_r": r})

    # Ranking por |r| y guardado a CSV
    corr_df = pd.DataFrame(rows).sort_values(by="pearson_r", key=lambda s: s.abs(), ascending=False)
    corr_df.to_csv(OUT_CSV, index=False)

    # Direccion en Z para pitch usando top-k
    pitch_corr_dir_z = build_corr_based_direction_in_Z(corr_df, k_total=k, topk=TOPK_FOR_Z_DIRS)

    # Guardar en NPZ sin perder claves anteriores
    existing = {}
    if OUT_NPZ.exists():
        existing = dict(np.load(OUT_NPZ, allow_pickle=True))
    existing["pitch_corr_dir_z"] = pitch_corr_dir_z
    np.savez(OUT_NPZ, **existing)

    # Mensajes de estado
    print(f"[OK] Guardado 'pitch_corr_dir_z' en {OUT_NPZ.name}")
    print(f"[OK] Correlaciones: {OUT_CSV.name}")

# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
