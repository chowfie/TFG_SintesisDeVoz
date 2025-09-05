# ============================================================
# CÓMO EJECUTAR (ejemplos)
# ------------------------------------------------------------
#
# Windows (PowerShell/CMD) — Common Voice (ES):
#   python .\<este_archivo>.py ^
#       --root "C:\datasets\CommonVoice_ES_1000" ^
#       --out_dir ".\latent_out\xtts_embeddings" ^
#       --exts ".wav,.flac,.mp3" ^
#       --min_seconds 1.5 ^
#       --max_utts_per_spk 10 ^
#       --corpus_name "CommonVoice-ES" ^
#       --save_gpt_latent
#
# Windows — VCTK:
#   python .\<este_archivo>.py ^
#       --root "C:\datasets\VCTK-Corpus-0.92\wav48" ^
#       --out_dir ".\latent_out\vctk_xtts" ^
#       --exts ".wav" ^
#       --max_utts_per_spk 12 ^
#       --corpus_name "VCTK"
#
# Windows — LibriTTS-R:
#   python .\<este_archivo>.py ^
#       --root "C:\datasets\LibriTTS_R" ^
#       --out_dir ".\latent_out\librittsr_xtts" ^
#       --exts ".wav,.flac" ^
#       --max_utts_per_spk 8 ^
#       --corpus_name "LibriTTS-R"
#
# Linux/macOS/WSL — Common Voice (ES):
#   python3 ./<este_archivo>.py \
#       --root "/datasets/CommonVoice_ES_1000" \
#       --out_dir "./latent_out/xtts_embeddings" \
#       --exts ".wav,.flac,.mp3" \
#       --min_seconds 1.5 \
#       --max_utts_per_spk 10 \
#       --corpus_name "CommonVoice-ES" \
#       --save_gpt_latent
#
# Salidas:
#   <out_dir>/embeddings_xtts_per_utterance.csv
#   <out_dir>/embeddings_xtts_per_speaker.csv
#
# ============================================================

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
from TTS.api import TTS  

# ------------------------------------------------------------
# Funciones auxiliares
# ------------------------------------------------------------

def list_audio_files(root: Path, exts: Tuple[str, ...]) -> List[Path]:
    """
        Busca todos los archivos de audio en una carpeta y sus subcarpetas.
        Root: carpeta principal donde buscar.
        Exts: tupla de extensiones a buscar (".wav",".mp3",".flac"...).
        Devuelve una lista de rutas completas.
    """
    files = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))  # Busca de forma recursiva
    return sorted(files)

def rel_parts(p: Path, root: Path) -> Tuple[str, ...]:
    """
        Devuelve las partes de la ruta de un archivo en forma de tupla de strings.
        Ejemplo:
          root = /datasets/LibriTTS_R
          p    = /datasets/LibriTTS_R/train-clean-100/19/198/file.flac
          salida -> ("train-clean-100", "19", "198", "file.flac")
    """
    try:
        # Si no está dentro de root, devolvemos la ruta tal cual
        rel = p.relative_to(root)
    except Exception:
        rel = p
    # parts divide la ruta en elementos separados
    return rel.parts


# Splits típicos de LibriTTS y LibriTTS-R
LIBRITTS_SPLITS = {
    "train-clean-100", "train-clean-360", "train-other-500",
    "dev-clean", "dev-other", "test-clean", "test-other"
}

def _looks_like_speaker_folder(s: str) -> bool:
    """
        Devuelve True si la cadena 's' parece ser un identificador de locutor.

        Reglas:
          - Solo dígitos (ej. "198") -> LibriTTS
          - Empieza con 'p' + dígitos (ej. "p225") -> VCTK
          - Cadena larga (>=8 chars, ej. "a9f7b23c") -> Common Voice
    """
    return s.isdigit() or (len(s) > 1 and s[0].lower() == "p" and s[1:].isdigit()) or len(s) >= 8

def _guess_corpus_name(root: Path) -> str:
    """
    Intenta identificar automáticamente el corpus según el nombre de la carpeta raíz.
    Ejemplo:
      - "CommonVoice_ES_1000" -> "CommonVoice-ES"
      - "LibriTTS_R"          -> "LibriTTS-R"
      - "VCTK-Corpus-0.92"    -> "VCTK"
    Si no reconoce nada, devuelve root.name tal cual.
    """
    r = root.name.lower()
    if "common" in r and "voice" in r:
        return "CommonVoice-ES"
    if "libritts" in r:
        return "LibriTTS-R"
    if "vctk" in r:
        return "VCTK"
    return root.name  # fallback si no hay coincidencia


def infer_ids(path: Path, root: Path, corpus_name: Optional[str] = None) -> Tuple[str, str]:
    """
    Dada la ruta de un audio, intenta adivinar:
      - corpus: nombre del dataset
      - speaker_id: carpeta que corresponde al locutor

    Soporta estructuras comunes:
      - LibriTTS:   split/speaker/chapter/file
      - CommonVoice: clips/speaker/file
      - VCTK:      wav*/speaker/file  o  speaker/file
      - Genérico: busca cualquier carpeta que parezca un speaker
    """
    parts = rel_parts(path, root)

    # Si no hay partes (ruta vacía o no relativa), devolvemos corpus + carpeta padre
    if len(parts) == 0:
        c = corpus_name if corpus_name else _guess_corpus_name(root)
        return c, path.parent.name

    # Determina corpus (si no se pasa a mano, lo adivina)
    corpus = corpus_name if corpus_name else _guess_corpus_name(root)
    low_root = corpus.lower()

    # --- Caso LibriTTS ---
    if "libritts" in low_root:
        # Estructura típica: [split, speaker, chapter, file]
        if len(parts) >= 2 and parts[0] in LIBRITTS_SPLITS:
            return corpus, parts[1]
        # Fallback: busca un folder que parezca speaker en cualquier nivel
        for p in parts[:-1]:
            if _looks_like_speaker_folder(p):
                return corpus, p
        # Último recurso: carpeta padre
        return corpus, parts[-2] if len(parts) >= 2 else path.parent.name

    # --- Caso Common Voice ---
    if "commonvoice" in low_root or ("common" in low_root and "voice" in low_root):
        # Speaker justo después de 'clips'
        if "clips" in parts:
            idx = parts.index("clips")
            if len(parts) >= idx + 2:
                return corpus, parts[idx + 1]
        return corpus, path.parent.name

    # --- Caso VCTK ---
    if "vctk" in low_root:
        # Estructura típica: wav*/speaker/file
        if len(parts) >= 2 and parts[0].startswith("wav"):
            return corpus, parts[1]
        # O directamente speaker/file
        if len(parts) >= 1:
            return corpus, parts[0]
        return corpus, path.parent.name

    # --- Caso genérico ---
    for p in reversed(parts[:-1]):  # recorremos carpetas intermedias
        if _looks_like_speaker_folder(p):
            return corpus, p

    # --- Último recurso ---
    speaker_id = parts[-2] if len(parts) >= 2 else path.parent.name
    return corpus, speaker_id

def is_long_enough(wav_path: Path, min_seconds: float = 1.5) -> bool:
    """
    Comprueba si un archivo de audio tiene al menos 'min_seconds' segundos de duración.
    Esto filtra audios demasiado cortos que no aportan información suficiente.
    """
    try:
        # Forma rápido con librosa
        dur = librosa.get_duration(path=str(wav_path))
        return dur >= min_seconds
    except Exception:
        try:
            # Fallback: cálculo manual con soundfile
            data, sr = sf.read(str(wav_path), always_2d=False)
            dur = len(data) / float(sr)
            return dur >= min_seconds
        except Exception:
            # Si ni siquiera se puede leer, descartamos
            return False

def ensure_mono_22050(in_path: Path, tmp_dir: Path) -> Path:
    """
    Convierte un audio al formato estándar que usa XTTS:
      - mono (1 canal)
      - 22.05 kHz de muestreo
    Guarda el archivo convertido en tmp_dir.
    Si falla la conversión, devuelve la ruta original sin cambios.
    """
    try:
        # Cargar y remuestrear con librosa (mono y sr=22050)
        y, sr = librosa.load(str(in_path), sr=22050, mono=True)

        # Ruta de salida en carpeta temporal
        out_path = tmp_dir / (in_path.stem + "_22050mono.wav")

        # Guardar como WAV PCM16, que es ligero y compatible
        sf.write(str(out_path), y, 22050, subtype="PCM_16")

        return out_path
    except Exception:
        # Si no se puede convertir, devolvemos el original
        return in_path


def write_csv_header(csv_path: Path, dim: int):
    """
    Crea un archivo CSV nuevo e introduce la cabecera.

    Columnas:
      - corpus: nombre del dataset
      - speaker_id: identificador del locutor
      - audio_path: ruta al archivo original
      - e0..e{dim-1}: valores del embedding de 512 dimensiones
    """
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Generamos nombres e0, e1, ..., e511 según 'dim'
        header = ["corpus", "speaker_id", "audio_path"] + [f"e{i}" for i in range(dim)]
        w.writerow(header)  # Escribimos la fila de cabecera


def append_csv_row(csv_path: Path, row: List):
    """
    Añade una fila de datos al CSV ya creado.

    Parámetros:
      csv_path: ruta del archivo CSV
      row: lista con los valores (corpus, speaker_id, audio_path, e0...e511)
    """
    # Abrimos en modo 'a' (append) para añadir al final sin borrar lo anterior
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(row)


def aggregate_per_speaker(per_utt_csv: Path, per_spk_csv: Path, dim: int):
    """
    A partir del CSV por audio (utterance), crea un CSV por locutor.

    Pasos:
      1. Lee todos los embeddings individuales.
      2. Agrupa por (corpus, speaker_id).
      3. Calcula la media de los 512 valores de embedding de cada locutor.
      4. Añade 'num_utts' con el número de audios usados por locutor.
      5. Guarda el resultado en per_spk_csv.
    """
    # Leemos el CSV de audios individuales
    df = pd.read_csv(per_utt_csv)

    # Columnas de embedding e0...e511
    cols = [f"e{i}" for i in range(dim)]

    # Media de los embeddings por locutor
    agg = df.groupby(["corpus", "speaker_id"], as_index=False)[cols].mean()

    # Añadimos columna con número de audios por speaker
    agg.insert(2, "num_utts", df.groupby(["corpus", "speaker_id"]).size().values)

    # Guardamos el CSV por speaker
    agg.to_csv(per_spk_csv, index=False)


# ------------------------------------------------------------
# Función principal
# ------------------------------------------------------------
def main():

    # Lee las opciones que pongamos en la terminal al ejecutar
    parser = argparse.ArgumentParser(description="Saca el embedding de voz 512D de XTTS por audio y por locutor.")
    parser.add_argument("--root", type=str, required=True, help="Carpeta donde están los audios del corpus (por ejemplo, CommonVoice_ES_1000 o LibriTTS_R)")
    parser.add_argument("--out_dir", type=str, required=True, help="Carpeta donde se guardarán los resultados")
    parser.add_argument("--exts", type=str, default=".wav,.flac,.mp3", help="Extensiones de audio que se van a buscar")
    parser.add_argument("--min_seconds", type=float, default=1.5, help="Tiempo mínimo de audio para usarlo")
    parser.add_argument("--max_utts_per_spk", type=int, default=10, help="Máximo de audios por locutor")
    parser.add_argument("--save_gpt_latent", action="store_true", help="Si se indica, guarda también el gpt_cond_latent de cada audio")
    parser.add_argument("--corpus_name", type=str, default=None, help="Etiqueta fija para el nombre de corpus (recomendado). Ej: 'CommonVoice-ES', 'LibriTTS-R', 'VCTK'")
    args = parser.parse_args()

    # Prepara las carpetas donde se va a trabajar
    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    tmp_dir = out_dir / "tmp_wavs"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Rutas de salida de resultados por audio y por speaker
    per_utt_csv = out_dir / "embeddings_xtts_per_utterance.csv"
    per_spk_csv = out_dir / "embeddings_xtts_per_speaker.csv"
    dim = 512 # tamaño del embedding de XTTS


    # Crear CSV por audio con cabecera e0..e511
    write_csv_header(per_utt_csv, dim)

    # Carga modelo XTTS
    print("Cargando modelo XTTS...")
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    model = tts.synthesizer.tts_model

    # Localizar audios a procesar
    exts = tuple([e.strip() for e in args.exts.split(",") if e.strip()])
    all_files = list_audio_files(root, exts)
    if not all_files:
        print("No se encontraron audios en", root, "con extensiones", exts)
        sys.exit(1)

    # Límite por locutor: contador de utterances ya procesadas
    per_spk_count: Dict[Tuple[str, str], int] = {}

    # Recorre y procesa cada audio encontrado
    for audio_path in tqdm(all_files, desc="Procesando audios"):
        try:
            # Salta audios demasiado cortos
            if not is_long_enough(audio_path, args.min_seconds):
                continue

            # Saca el corpus y el ID del locutor
            corpus, speaker_id = infer_ids(audio_path, root, args.corpus_name)
            key = (corpus, speaker_id)

            # Si ya hemos llegado al máximo para este locutor, saltamos
            n_done = per_spk_count.get(key, 0)
            if n_done >= args.max_utts_per_spk:
                continue

            # Pasa el audio a formato mono 22.05 kHz
            norm_wav = ensure_mono_22050(audio_path, tmp_dir)

            # Obtiene dos cosas del modelo:
            # - gpt_cond_latent: información para el generador
            # - speaker_embedding: huella de voz del locutor
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                audio_path=[str(norm_wav)]
            )

            # Aplanar a vector 512D numpy
            e = speaker_embedding.squeeze().detach().cpu().numpy().reshape(-1)
            if e.shape[0] != dim:
                # Si por lo que sea no es 512, saltamos este archivo
                continue

            # Escribir fila en el CSV por audio
            row = [corpus, speaker_id, str(audio_path)] + [float(x) for x in e]
            append_csv_row(per_utt_csv, row)

            # Guardar opcionalmente gpt_cond_latent
            if args.save_gpt_latent:
                gpt_np = gpt_cond_latent.squeeze().detach().cpu().numpy()
                gpt_dir = out_dir / "gpt_latents" / corpus / speaker_id
                gpt_dir.mkdir(parents=True, exist_ok=True)
                np.save(gpt_dir / (audio_path.stem + "_gpt.npy"), gpt_np)

            # Actualizamos el número de audios usados para este locutor
            per_spk_count[key] = n_done + 1

        except Exception as e:
            print(f"[WARN] Fallo con {audio_path}: {e}")
            continue

    # Resumen por locutor
    try:
        # Promedia e0..e511 por (corpus, speaker_id) y añade num_utts
        aggregate_per_speaker(per_utt_csv, per_spk_csv, dim)
        print("Guardado:", per_utt_csv)
        print("Guardado:", per_spk_csv)
    except Exception as e:
        print("[WARN] No se pudo agregar por speaker:", e)


if __name__ == "__main__":
    main()
