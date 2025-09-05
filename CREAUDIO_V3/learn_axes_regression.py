# python learn_axes_regression.py

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# ------------------------------------------------------------
# Rutas
# ------------------------------------------------------------
LATENT_DIR = Path("./latent_out")
EMB_CSV   = LATENT_DIR / "embeddings_with_metadata.csv"
PITCH_CSV = LATENT_DIR / "pitch_per_speaker.csv"
OUT_NPZ   = LATENT_DIR / "semantic_directions_regression.npz"

# ------------------------------------------------------------
# Funciones auxiliares
# ------------------------------------------------------------
def unit(v, eps=1e-9):
    """Normaliza un vector a norma L2 = 1."""
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return (v / max(n, eps)).astype(np.float32)

def load_embeddings():
    """
       Carga el archivo de embeddings con metadatos

       - Lee el CSV completo en un DataFrame (df)
       - Detecta las columnas que empiezan por 'e'
       - Extrae esas columnas como una matriz numérica (X) de floats
       - Devuelve:
           * df → DataFrame completo con metadatos (AGE, GENDER, CORPUS…)
           * X  → matriz numpy con los embeddings (filas=locutores, columnas=dimensiones)
           * emb_cols → lista de nombres de las columnas de embedding
       """
    df = pd.read_csv(EMB_CSV, low_memory=False)
    emb_cols = [c for c in df.columns if c.startswith("e")]
    X = df[emb_cols].to_numpy(dtype=np.float32)
    return df, X, emb_cols

def map_gender(series):
    """Convierte GENDER a binario: M=1, F=0."""
    s = series.astype(str).str.upper().replace({
        "MALE":"M","FEMALE":"F","HOMBRE":"M","MUJER":"F"
    })
    y = (s == "M").astype(float)
    return y

# ------------------------------------------------------------
# Ajuste de regresión en 512D
# ------------------------------------------------------------
def fit_direction_ridge_512(X, y, corpus=None, alpha=1.0):
    """
        Ajusta una regresión ridge y devuelve la dirección w en 512D
        Esa dirección nos dice hacia dónde movernos si queremos que cambie una
        característica de la voz (ejemplo: más masculino, más mayor, más agudo...)
            - X: todos los embeddings
            - y: los valores que queremos aprender (0/1 para hombre/mujer, edad en años,
                 o pitch medio)
            - corpus: opcional, de qué base de datos viene cada persona
            - alpha: un número que hace que la regresión sea más estable
    """

    # Filtramos filas válidas
    mask = np.isfinite(y)
    if corpus is not None:
        mask &= corpus.notna().to_numpy()

    # Aplicamos la máscara a X e y
    Xf = X[mask]
    yf = y[mask]

    # Si quedan muy pocos datos, devolvemos vacío
    if Xf.shape[0] < 10:
        return None, np.nan

    # Pipeline: escalado de X + regresión ridge
    pipe = make_pipeline(
        # Ajustar la escala de los datos para que todas las columnas tengan más o menos la misma importancia
        StandardScaler(with_mean=True, with_std=True),
        # Entrenar una regresión ridge (una línea/hiperplano que intenta predecir y a partir de X)
        Ridge(alpha=alpha, fit_intercept=True)
    )
    pipe.fit(Xf, yf) # entrenamos con los datos

    # Recuperamos pesos de la regresión en el espacio original
    scaler = pipe.named_steps["standardscaler"]
    ridge  = pipe.named_steps["ridge"]

    # Coeficientes en el espacio escalado
    w_scaled = ridge.coef_.astype(np.float32)
    # Reescalamos a espacio original (dividimos por la desviación de cada dim)
    w_original = w_scaled / np.maximum(scaler.scale_.astype(np.float32), 1e-9)
    # Normalizamos la dirección a norma 1
    w = unit(w_original)

    # Calculamos R^2 para medir calidad del ajuste
    y_pred = pipe.predict(Xf)
    score = r2_score(yf, y_pred)

    # Devolvemos la dirección aprendida y el R^2
    return w, score


# ------------------------------------------------------------
# Unión de pitch externo
# ------------------------------------------------------------
def attach_pitch(df_emb: pd.DataFrame) -> pd.DataFrame:
    """
        Une el pitch medio por locutor desde pitch_per_speaker.csv al
        DataFrame de embeddings. Devuelve un df con columna LOG_F0 lista
          - Buscamos clave común (speaker_id o speaker_uid)
          - Si el CSV externo ya trae LOG_F0, lo usamos
          - Si trae F0 en Hz (F0_HZ / MEAN_F0_HZ / F0), calculamos LOG_F0 = log(F0)
          - Hacemos un 'merge' para añadir LOG_F0 a nuestro df principal
    """
    # Si no existe el CSV de pitch, no hacemos nada
    if not PITCH_CSV.exists():
        print(f"[INFO] No encuentro {PITCH_CSV}.")
        return df_emb

    # Cargamos el CSV de pitch
    df_pitch = pd.read_csv(PITCH_CSV, low_memory=False)

    # Detectar clave de cruce (speaker_id)
    key = None
    if "speaker_id" in df_emb.columns and "speaker_id" in df_pitch.columns:
        key = "speaker_id"
    elif "speaker_uid" in df_emb.columns and "speaker_uid" in df_pitch.columns:
        key = "speaker_uid"

    # Si no hay clave común, no podemos unir
    if key is None:
        print("[WARN] No encuentro clave común (speaker_id o speaker_uid). Salto unión de pitch.")
        return df_emb

    # Detectar columna de pitch
    pitch_col = None
    cand_pitch = ["LOG_F0", "F0_HZ", "MEAN_F0_HZ", "F0"]
    for c in cand_pitch:
        if c in df_pitch.columns:
            pitch_col = c
            break

    # Si no hay ninguna columna reconocible de pitch, salimos
    if pitch_col is None:
        print("[WARN] No hay columna de pitch compatible en el CSV externo.")
        return df_emb

    # Nos quedamos con la clave + columna de pitch
    dfp = df_pitch[[key, pitch_col]].copy()

    # 5) Aseguramos que exista LOG_F0:
    #    - Si ya es LOG_F0: lo convertimos a numérico
    #    - Si viene en Hz: pasamos a LOG_F0 = log(F0) con un mínimo para evitar log(0)
    if pitch_col == "LOG_F0":
        dfp["LOG_F0"] = pd.to_numeric(dfp["LOG_F0"], errors="coerce")
    else:
        f0 = pd.to_numeric(dfp[pitch_col], errors="coerce")
        f0 = f0.clip(lower=1e-6)
        dfp["LOG_F0"] = np.log(f0)

    # Unir al df de embeddings
    df_out = df_emb.merge(dfp[[key, "LOG_F0"]], on=key, how="left", validate="m:1")
    return df_out

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():

    # Cargamos CSV
    df, X, emb_cols = load_embeddings()

    # Detectar corpus
    corpus_col = None
    if "CORPUS" in df.columns:
        corpus_col = df["CORPUS"]
    elif "corpus" in df.columns:
        corpus_col = df["corpus"]

    out = {}

    # ------ GÉNERO ------
    if "GENDER" in df.columns:
        # Pasamos la columna a un vector 0/1 (F=0, M=1)
        y_gender = map_gender(df.get("GENDER"))
        # Entrenamos
        w_gender, r2_gender = fit_direction_ridge_512(X, y_gender, corpus=corpus_col, alpha=1.0)
        # Guardamos y mostramos R^2
        if w_gender is not None:
            out["gender_reg_512"] = w_gender
            out["meta_regression_r2_gender"] = np.array([r2_gender], dtype=np.float32)
            print(f"[OK] Eje género entrenado. R^2 = {r2_gender:.3f}")
    else:
        print("[WARN] No hay columna GENDER")

    # ------ EDAD ------
    if "AGE" in df.columns:
        # Convertimos la columna AGE a números
        y_age = pd.to_numeric(df.get("AGE"), errors="coerce").to_numpy(dtype=np.float32)
        # Entrenamos
        w_age, r2_age = fit_direction_ridge_512(X, y_age, corpus=corpus_col, alpha=5.0)
        # Guardamos y mostramos R^2
        if w_age is not None:
            out["age_reg_512"] = w_age
            out["meta_regression_r2_age"] = np.array([r2_age], dtype=np.float32)
            print(f"[OK] Eje edad entrenado. R^2 = {r2_age:.3f}")
    else:
        print("[WARN] No hay columna AGE")

    # ------ TONO ------
    df_with_pitch = attach_pitch(df)
    if "LOG_F0" in df_with_pitch.columns:
        y_pitch = pd.to_numeric(df_with_pitch["LOG_F0"], errors="coerce").to_numpy(dtype=np.float32)
        ok = np.isfinite(y_pitch)
        if ok.sum() >= 20:
            w_pitch, r2_pitch = fit_direction_ridge_512(
                X[ok], y_pitch[ok], corpus=(corpus_col[ok] if corpus_col is not None else None), alpha=5.0
            )
            if w_pitch is not None:
                out["pitch_reg_512"] = w_pitch
                out["meta_regression_r2_pitch"] = np.array([r2_pitch], dtype=np.float32)
                print(f"[OK] Eje pitch entrenado. R^2 = {r2_pitch:.3f}")
        else:
            print("[INFO] Muy pocos LOG_F0 válidos")
    else:
        print("[INFO] No se pudo crear LOG_F0 tras la unión")

    # Guardar
    old = {}
    if OUT_NPZ.exists():
        old = dict(np.load(OUT_NPZ, allow_pickle=True))
    old.update(out)
    old["meta_regression_notes"] = "Ridge 512D; direcciones normalizadas; corpus opcional; pitch unido desde pitch_per_speaker.csv"
    np.savez(OUT_NPZ, **old)

    print("Guardado:", OUT_NPZ)
    print("Claves guardadas:", list(out.keys()))

if __name__ == "__main__":
    main()
