import pandas as pd
from pathlib import Path
import re
import sys

# ------------------------------------------------------------
# Rutas
# ------------------------------------------------------------
CV_SPK       = Path(r"C:\Users\sreal\Desktop\TFG\CODIGO\CREAUDIO_V3\embeddings_cv\embeddings_cv_per_speaker.csv")
VCTK_SPK     = Path(r"C:\Users\sreal\Desktop\TFG\CODIGO\CREAUDIO_V3\embeddings_vctk\embeddings_vctk_per_speaker.csv")
LIBRITTS_SPK = Path(r"C:\Users\sreal\Desktop\TFG\CODIGO\CREAUDIO_V3\embeddings_librittsr\embeddings_libritts_per_speaker.csv")

OUT_DIR  = Path(r"C:\Users\sreal\Desktop\TFG\CODIGO\CREAUDIO_V3\merged_all")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_SPK  = OUT_DIR / "merged_per_speaker.csv"

# ------------------------------------------------------------
# Funciones auxiliares
# ------------------------------------------------------------
def _emb_cols(df: pd.DataFrame):
    """
    Devuelve la lista de columnas de embedding con nombre 'e0', 'e1', ..., 'eN'
    ordenadas numéricamente por su índice.
    """
    # Filtra columnas cuyo nombre sea 'e' + dígitos (regex exacta)
    cols = [c for c in df.columns if re.fullmatch(r"e\d+", c)]
    # Ordena por el número que hay tras la 'e'
    return sorted(cols, key=lambda c: int(c[1:]))

def _ensure_cols(df: pd.DataFrame, need, name: str):
    """
    Verifica que contenga todas las columnas de la lista 'need'
    Si falta alguna, lanza un ValueError con un mensaje indicando qué dataset ('name')
    y qué columnas no estaban presentes.
    """
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Faltan columnas: {missing}")


def load_with_corpus(path: Path, corpus_name: str) -> pd.DataFrame:
    """
    Carga un CSV de embeddings y lo normaliza con información de corpus.

    Pasos:
      1. Comprueba que el archivo exista y lo lee en DataFrame.
      2. Detecta columnas de embedding e0...eN y valida que exista speaker_id.
      3. Asegura que la columna 'corpus' tenga siempre el valor corpus_name.
      4. Añade/actualiza 'speaker_uid' = corpus_name:speaker_id (identificador único).
      5. Conserva solo columnas relevantes: corpus, speaker_uid, speaker_id,
         num_utts (si existe) y los embeddings.
      6. Devuelve el DataFrame listo para concatenar con otros corpus.
    """
    print(f"\n[LOAD] {corpus_name}: {path}")
    if not path.exists():
        raise FileNotFoundError(f"No encuentro {path}")

    # Leemos el CSV
    df = pd.read_csv(path, low_memory=False)
    print(f"[{corpus_name}] shape: {df.shape}")

    # Detectar columnas de embedding y validar
    emb_cols = _emb_cols(df)
    print(f"[{corpus_name}] columnas embedding detectadas: {len(emb_cols)} (ej: {emb_cols[:6]}...)")
    if not emb_cols:
        raise ValueError(f"[{corpus_name}] {path} no tiene columnas e0..eN")
    if "speaker_id" not in df.columns:
        raise ValueError(f"[{corpus_name}] {path} no tiene columna speaker_id")

    # Forzar columna 'corpus' homogénea
    if "corpus" in df.columns:
        df["corpus"] = corpus_name
    else:
        df.insert(0, "corpus", corpus_name)

    # Crear identificador único corpus:speaker
    if "speaker_uid" not in df.columns:
        df.insert(1, "speaker_uid", df["speaker_id"].apply(lambda s: f"{corpus_name}:{s}"))
    else:
        df["speaker_uid"] = df["speaker_id"].apply(lambda s: f"{corpus_name}:{s}")

    # Seleccionar columnas finales
    keep = ["corpus", "speaker_uid", "speaker_id"]
    if "num_utts" in df.columns:
        keep.append("num_utts")
    keep += emb_cols

    # Copiar solo las columnas necesarias
    out = df[keep].copy()
    print(f"[{corpus_name}] preview:\n{out.head(2)}")
    return out

# ------------------------------------------------------------
# Función principal
# ------------------------------------------------------------
def main():
    print("Cargando CSV por speaker...")

    # Cargar cada dataset normalizado con la etiqueta de corpus
    cv   = load_with_corpus(CV_SPK,       "CommonVoice-ES")
    vctk = load_with_corpus(VCTK_SPK,     "VCTK")
    lib  = load_with_corpus(LIBRITTS_SPK, "LibriTTS-R")

    # Validación rápida de que las cargas tienen sentido
    #    - Debe haber filas
    #    - Debe haber suficientes columnas e* (los 512 de embeddings)
    for name, df in [("CommonVoice-ES", cv), ("VCTK", vctk), ("LibriTTS-R", lib)]:
        if len(df) == 0:
            raise ValueError(f"[{name}] No hay filas tras la carga. Revisa la ruta o el CSV.")
        if len(_emb_cols(df)) < 10:  # umbral blando para detectar CSV erróneo
            raise ValueError(f"[{name}] Muy pocas columnas e*. ¿Seguro que es el CSV correcto?")

    # Unificar en un solo DataFrame
    merged = pd.concat([cv, vctk, lib], ignore_index=True)

    # Eliminar posibles duplicados por speaker_uid
    #    (por ejemplo, si un mismo speaker aparece dos veces por algún motivo)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["speaker_uid"])
    after = len(merged)

    # 5) Guardar resultado final
    merged.to_csv(OUT_SPK, index=False)

    # 6) Resumen por consola
    emb_cols = _emb_cols(merged)
    print("\nResumen")
    print(" - columnas embedding:", len(emb_cols))
    print(" - total speakers:", after, f"(eliminados {before - after} duplicados)")
    print(merged["corpus"].value_counts())
    print(f"\nListo: {OUT_SPK}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR]", e)
        sys.exit(1)
