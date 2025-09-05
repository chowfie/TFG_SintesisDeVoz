# python learn_axes.py

import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from joblib import load
import json

# ------------------------------------------------------------
# Configuración: rutas y parámetros por defecto
# ------------------------------------------------------------

# CSV principal de embeddings ya unificados (por speaker)
INPUT_CSV_CANDIDATES = [Path("C:\\Users\\sreal\\Desktop\\TFG\\CODIGO\\CREAUDIO_V3\\merged_all\\merged_per_speaker.csv")]
# Metadatos opcionales
VCTK_INFO = Path("C:\\Users\\sreal\\Desktop\\TFG\\CODIGO\\datasets\\VCTK-Corpus-0.92\\speaker-info.txt")
CV_VALIDATED = Path("C:\\Users\\sreal\\Desktop\\TFG\\CODIGO\\datasets\\CommonVoice_ES_1000\\validated.tsv")
LIB_SPK_TSV = Path("C:\\Users\\sreal\\Desktop\\TFG\\CODIGO\\datasets\\LibriTTS_R\\train-clean-100\\speakers.tsv")

# Salida
OUT_DIR = Path("./latent_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Nº de componentes PCA que queremos (sliders "conceptuales")
N_COMPONENTS = 5

# Nº de ejes top por |r| que se usarán para construir una dirección en Z basada en correlaciones.
TOPK_FOR_Z_DIRS = 3

# ------------------------------------------------------------
# Funciones auxiliares
# ------------------------------------------------------------
def _emb_cols(df: pd.DataFrame) -> List[str]:
    """
    Devuelve las columnas de embedding ('e0', 'e1', ...), ordenadas por su índice numérico.
    - Intenta primero el patrón estricto 'e' + dígitos (e.g., 'e0', 'e42').
    - Si no encuentra nada, acepta variantes como 'e_0' o 'e-0' (patrón alternativo).
    """
    cols = [c for c in df.columns if re.fullmatch(r"e\d+", c)]

    # Fallback: algunos CSV pueden venir como e_0 / e-0; los capturamos si no hubo match antes
    if not cols:
        alt = [c for c in df.columns if re.fullmatch(r"e[_-]?\d+", c)]
        if alt:
            cols = alt

    return sorted(cols, key=lambda c: int(re.findall(r"\d+", c)[0]))

def _unit(vec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Devuelve 'vec' normalizado a norma L2 = 1.
    - Útil para usar 'vec' como dirección (solo importa la orientación, no la escala).
    - 'eps' evita divisiones por cero cuando el vector es (casi) nulo.
    """
    n = float(np.linalg.norm(vec))
    return vec / max(n, eps)

def normalize_ids(emb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza los identificadores de locutores (speaker_id) según el corpus de origen.
    ¿Por qué hace falta?
      - Cada dataset usa un formato distinto para los IDs de los speakers.
      - Si no unificamos estos formatos, luego al juntar datos de varios corpus
        habría inconsistencias y problemas de identificación.

    Reglas aplicadas:
      - VCTK → siempre con prefijo "p" (ejemplo: "225" → "p225").
      - LibriTTS-R → asegurar que los IDs sean strings (no enteros).
      - CommonVoice → no se modifica (ya está correctos).

    Devuelve un DataFrame nuevo con los speaker_id normalizados.
    """

    # Hacemos una copia para no modificar el DataFrame original directamente
    df = emb_df.copy()

    # VCTK
    mask_vctk = df["corpus"] == "VCTK"
    df.loc[mask_vctk, "speaker_id"] = df.loc[mask_vctk, "speaker_id"].astype(str)
    df.loc[mask_vctk, "speaker_id"] = df.loc[mask_vctk, "speaker_id"].apply(
        lambda x: f"p{x}" if not str(x).startswith("p") else str(x)
    )

    # LibriTTS-R
    # En LibriTTS los speaker_id a veces son numéricos (ej: 1234).
    # Nos aseguramos de que sean siempre strings, no enteros.
    mask_lib = df["corpus"] == "LibriTTS-R"
    df.loc[mask_lib, "speaker_id"] = df.loc[mask_lib, "speaker_id"].astype(str)

    # CommonVoice
    # En CommonVoice no hay que hacer nada especial, los IDs ya están correctos.

    return df

def mode_non_null(series: pd.Series) -> Optional[str]:
    '''
        Moda por speaker_id para consolidar etiquetas (a veces en la edad o género hay discrepancias,
        y no todos los registros de un mismo locutor coinciden; aquí nos quedamos con el valor más frecuente).
    '''
    # Eliminamos valores nulos
    s = series.dropna()

    # Si la serie queda vacía tras quitar NaNs, devolvemos None
    if s.empty:
        return None

    # Calcular moda
    m = s.mode()

    # Si existe al menos un valor modal, devolvemos el primero.
    # Si por algún motivo la moda también está vacía, devolvemos simplemente el primer valor disponible.
    return m.iloc[0] if not m.empty else s.iloc[0]

def _cv_age_to_num(x: Optional[str]):
    """
    Convierte las etiquetas de edad de CommonVoice (texto) en un número representativo.
    - CommonVoice no da siempre la edad exacta, sino rangos o etiquetas como 'twenties', 'forties', etc.
    - Aquí hacemos un mapeo aproximado a un valor numérico (la media del rango).
      Ejemplo: "twenties" -> 25, "fifties" -> 55.
    - Si no se reconoce el valor, devuelve None.
    """
    # Si la entrada es None o NaN, devolvemos None
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None

    # Convertimos a string, quitamos espacios y pasamos a minúsculas para comparar
    t = str(x).strip().lower()

    # Diccionario de equivalencias texto -> valor
    table = {
        "teens": 15,
        "teen": 15,
        "twenties": 25, "twenty": 25, "twentyish": 25,
        "thirties": 35, "thirty": 35,
        "forties": 45, "forty": 45,
        "fifties": 55, "fifty": 55,
        "sixties": 65, "sixty": 65,
        "seventies": 75, "seventy": 75,
        "eighties": 85, "eighty": 85,
        "nineties": 95, "ninety": 95,
    }

    # Busca en la tabla; si no está, devuelve None
    return table.get(t, None)

# ------------------------------------------------------------
# Carga de datos: embeddings unificados + metadatos opcionales
# ------------------------------------------------------------
def load_merged_embeddings() -> pd.DataFrame:
    """
    Intenta cargar el CSV unificado por speaker ('merged_per_speaker.csv')

    Flujo:
      - Recorre las rutas candidatas en orden.
      - Si encuentra una que existe, lee el CSV y lo devuelve como DataFrame.
      - Si no encuentra ninguna, lanza FileNotFoundError (artefacto faltante del pipeline).
    """
    for p in INPUT_CSV_CANDIDATES:
        if p.exists():
            # Carga inmediata del CSV válido
            df = pd.read_csv(p)
            print(f"[OK] Cargado embeddings: {p}")
            return df

    # Ninguna ruta candidata existe → error explícito para depurar rápido
    raise FileNotFoundError("No se encontró 'merged_per_speaker.csv'")

def parse_vctk_speaker_info(path_txt: Path) -> pd.DataFrame:
    """
    Parsea el archivo de VCTK 'speaker-info.txt', donde cada línea describe un locutor.
    El archivo suele estar separado por varios espacios.
    Devuelve un DataFrame con columnas estándar para el merge de metadatos.
    """
    # Si el fichero no existe, devolver DF vacío con las columnas esperadas
    if not path_txt.exists():
        return pd.DataFrame(columns=["speaker_id","AGE","GENDER","ACCENT","REGION","corpus"])

    rows = []

    # Leer texto
    for line in path_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        # Saltar líneas vacías, de comentario o la fila de cabecera
        if not line or line.startswith("#") or line.startswith("ID"):
            continue

        # Separar usando 2 o más espacios como delimitador (las columnas vienen así)
        parts = re.split(r"\s{2,}", line)

        # Esperamos al menos: ID, AGE, GENDER, ACCENT; REGION
        if len(parts) >= 4:
            spk_id = parts[0]                 # p.ej., "p225"
            age = parts[1]                    # p.ej., "23"
            gender = parts[2]                 # "M" o "F"
            accents = parts[3]                # p.ej., "English"
            region = parts[4] if len(parts) >= 5 else None  # p.ej., "Southern England"

            # Guardar fila normalizada (AGE numérico; valores no numéricos -> NaN)
            rows.append({
                "speaker_id": spk_id,
                "AGE": pd.to_numeric(age, errors="coerce"),
                "GENDER": gender,
                "ACCENT": accents,
                "REGION": region
            })


    out = pd.DataFrame(rows)
    out["corpus"] = "VCTK"

    return out

def parse_commonvoice_validated(path_tsv: Path) -> pd.DataFrame:
    """
        CommonVoice validated.tsv: tiene info por grabación
        Normalizamos:
          - AGE (texto) -> número (midpoints)
          - GENDER -> M/F
          - ACCENT -> tal cual (si existe)
        Luego consolidamos por speaker_id con la moda.
    """
    # Si el archivo no existe, devolvemos un DataFrame vacío con las columnas esperadas
    if not path_tsv.exists():
        return pd.DataFrame(columns=["speaker_id","AGE","GENDER","ACCENT","REGION","corpus"])

    try:
        cv = pd.read_csv(path_tsv, sep="\t", dtype=str, encoding="utf-8", quoting=3, low_memory=False)
    except Exception:
        Exception("No se ha podido leer el TSV de CommonVoice")

    # ----- Renombrado de columnas a nombres estándar -----
    #   - client_id -> speaker_id
    #   - age       -> AGE
    #   - gender    -> GENDER
    #   - accent(s) -> ACCENT
    colmap = {}
    for c in cv.columns:
        lc = c.lower()
        if lc == "client_id": colmap[c] = "speaker_id"
        elif lc == "age":     colmap[c] = "AGE"
        elif lc == "gender":  colmap[c] = "GENDER"
        elif lc in ("accent", "accents"): colmap[c] = "ACCENT"
    cv = cv.rename(columns=colmap)

    # Nos quedamos con lo disponible entre estas
    keep = [c for c in ["speaker_id","AGE","GENDER","ACCENT"] if c in cv.columns]
    # Si no encontramos speaker_id, devolver vacio
    if "speaker_id" not in keep:
        return pd.DataFrame(columns=["speaker_id","AGE","GENDER","ACCENT","REGION","corpus"])

    # Reducimos a las columnas relevantes
    cv = cv[keep].copy()

    # ---- Normaliza AGE ----
    if "AGE" in cv.columns:
        # Aplica _cv_age_to_num fila a fila (twenties→25, thirties→35...)
        cv["AGE"] = cv["AGE"].apply(_cv_age_to_num)

    # ---- Normaliza GENDER ----
    if "GENDER" in cv.columns:
        # Limpieza de espacios/mayúsculas y normalización de variantes (“male”, “hombre”...)
        cv["GENDER"] = (cv["GENDER"].astype(str).str.strip().str.lower()
                        .replace({"male": "M", "m": "M", "hombre": "M",
                                  "female": "F", "f": "F", "mujer": "F"}))
        # Cualquier valor fuera de M/F se convierte en NaN para no contaminar correlaciones
        cv["GENDER"] = cv["GENDER"].where(cv["GENDER"].isin(["M", "F"]))

    # Agregamos por speaker_id aplicando la moda a cada columna meta disponible
    agg = cv.groupby("speaker_id", as_index=False).agg({c: mode_non_null for c in keep if c != "speaker_id"})

    # Aseguramos todas las columnas meta aunque falten
    for c in ["AGE","GENDER","ACCENT"]:
        if c not in agg.columns:
            agg[c] = None

    # Common Voice no trae región homogénea, rellenamos con None para mantener el esquema
    agg["REGION"] = None
    agg["corpus"] = "CommonVoice-ES"
    return agg

def parse_libritts_speakers(path_tsv: Path) -> pd.DataFrame:
    """
    LibriTTS-R speakers.tsv con columnas: READER, GENDER, SUBSET NAME.
    - READER: id del locutor (numérico) -> lo mapeamos a speaker_id (str)
    - GENDER: F/M -> lo normalizamos a M/F
    - No trae AGE/ACCENT/REGION -> NaN
    """
    # Si el archivo no existe, devolvemos un DataFrame vacío con las columnas estándar.
    if not path_tsv.exists():
        return pd.DataFrame(columns=["speaker_id","AGE","GENDER","ACCENT","REGION","corpus"])

    df = pd.read_csv(path_tsv, sep="\t", dtype=str, encoding="utf-8", quoting=3, low_memory=False)

    # Localizamos columnas (por si vienen con mayúsculas)
    cols = {c.lower(): c for c in df.columns}
    # Identificamos las columnas READER (id de locutor) y GENDER.
    reader_col = cols.get("reader")
    gender_col = cols.get("gender")

    # Sin columna READER no podemos unir por speaker_id
    if reader_col is None:
        # Sin READER no podemos unir por speaker_id
        return pd.DataFrame(columns=["speaker_id","AGE","GENDER","ACCENT","REGION","corpus"])

    out = pd.DataFrame({
        # speaker_id como string
        "speaker_id": df[reader_col].astype(str)
    })

    # Normalizamos la columna de género si existe.
    if gender_col is not None:
        g = (df[gender_col].astype(str).str.strip().str.upper()
             .replace({"MALE":"M","FEMALE":"F","M":"M","F":"F"}))
        out["GENDER"] = g.where(g.isin(["M","F"]))
    else:
        out["GENDER"] = np.nan

    # En LibriTTS no tenemos información de edad, acento o región: rellenamos con NaN.
    out["AGE"] = np.nan
    out["ACCENT"] = np.nan
    out["REGION"] = np.nan
    out["corpus"] = "LibriTTS-R"

    return out

def attach_metadata(emb_df: pd.DataFrame) -> pd.DataFrame:
    """Une los embeddings con los metadatos de cada corpus (si existen)."""
    metas = []
    # Columnas a conservar
    want = ["corpus","speaker_id","AGE","GENDER","ACCENT","REGION"]

    # Parseamos VCTK
    vctk = parse_vctk_speaker_info(VCTK_INFO)
    if not vctk.empty:
        metas.append(vctk.reindex(columns=want))

    # Parseamos CommonVoice
    cv = parse_commonvoice_validated(CV_VALIDATED)
    if not cv.empty:
        metas.append(cv.reindex(columns=want))

    # Parseamos LibriTTS-R (si existe speakers.tsv)
    lib = parse_libritts_speakers(LIB_SPK_TSV)
    if not lib.empty:
        metas.append(lib.reindex(columns=want))

    # Si no hemos conseguido cargar ningún metadato, devolvemos embeddings con columnas vacías para AGE/GENDER/ACC
    if not metas:
        df = emb_df.copy()
        for c in ["AGE","GENDER","ACCENT","REGION"]:
            if c not in df.columns:
                df[c] = np.nan
        return df

    # Concatenamos todos los DataFrames de metadatos
    meta = pd.concat(metas, ignore_index=True)
    # Unimos por "corpus" y "speaker_id"
    out = emb_df.merge(meta, on=["corpus","speaker_id"], how="left")
    return out

# ------------------------------------------------------------
# PCA, correlaciones y direcciones semánticas
# ------------------------------------------------------------
def correlations(Z: np.ndarray, df_meta: pd.DataFrame):
    """
        Calcula la correlación de cada componente PCA (z1, z2, …) con la edad y el género.
        - Edad → correlación de Pearson con los valores numéricos de AGE.
        - Género → pasamos M/F a 1/0 y hacemos también correlación de Pearson.
        Devuelve dos DataFrames: uno con correlaciones con la edad y otro con correlaciones con el género.

        Nota:
        La correlación de Pearson mide el grado de relación lineal entre dos variables.
        Su valor va de -1 a 1:
          *  1  → relación lineal positiva perfecta (si X sube, Y sube).
          * -1 → relación lineal negativa perfecta (si X sube, Y baja).
          *  0  → no hay relación lineal.
    """

    # Creamos nombres más amigables para las columnas de Z: z1, z2, ..., zk
    # (por defecto PCA da índices numéricos, aquí los renombramos para que sea más claro)
    cols = [f"z{i+1}" for i in range(Z.shape[1])]
    Zdf = pd.DataFrame(Z, columns=cols)

    # ---------- CORRELACIÓN EDAD ----------
    # Convertimos la columna AGE a numérico
    age = pd.to_numeric(df_meta.get("AGE"), errors="coerce")
    age_corr = []

    # Para cada componente z_i calculamos la correlación de Pearson con la edad
    for c in cols:
        # pd.concat alinea por índice; corr() calcula la matriz de correlaciones 2x2
        r = pd.concat([Zdf[c], age], axis=1).corr(method="pearson").iloc[0, 1]
        age_corr.append({"component": c, "pearson_r": r})

    # Creamos DataFrame ordenado por valor absoluto de correlación
    age_corr_df = pd.DataFrame(age_corr).sort_values(
        by="pearson_r", key=lambda s: s.abs(), ascending=False
    )

    # ---------- CORRELACIÓN GÉNERO ----------
    # Normalizamos textos a "M"/"F"
    g = df_meta.get("GENDER").astype(str).str.upper().replace({
        "MALE": "M", "FEMALE": "F", "HOMBRE": "M", "MUJER": "F"
    })

    # Cualquier valor que no sea M/F lo ponemos como NaN
    g = g.where(g.isin(["M", "F"]), np.nan)

    # Mapeamos a numérico: M=1.0, F=0.0 para poder hacer Pearson
    g_num = g.map({"M": 1.0, "F": 0.0})

    gender_corr = []
    # Igual que con edad, recorremos cada eje y sacamos correlación con el género numérico
    for c in cols:
        r = pd.concat([Zdf[c], g_num], axis=1).corr(method="pearson").iloc[0, 1]
        gender_corr.append({"component": c, "pearson_r": r})

    # Ordenamos por correlación absoluta para ver qué componentes diferencian más el género
    gender_corr_df = pd.DataFrame(gender_corr).sort_values(
        by="pearson_r", key=lambda s: s.abs(), ascending=False
    )

    return age_corr_df, gender_corr_df

def semantic_directions(df_meta: pd.DataFrame, X: np.ndarray) -> dict:
    """
        Calcula direcciones "semánticas" en el espacio original de embeddings (512D).
        La idea es obtener vectores que representen diferencias medias entre grupos:
          - Hombres vs Mujeres
          - Mayores vs Jóvenes (Q4 - Q1 en edad)
        Devuelve un diccionario con las direcciones normalizadas (norma L2=1).
    """
    out = {}

    # ---------- DIRECCIÓN DE GENERO ----------
    # Normalizamos los valores de la columna GENDER a "M" o "F"
    g = df_meta.get("GENDER").astype(str).str.upper().replace({
        "MALE": "M", "FEMALE": "F", "HOMBRE": "M", "MUJER": "F"
    })

    # Creamos máscaras booleanas para hombres y mujeres
    mask_m = g == "M"
    mask_f = g == "F"
    # Solo calculamos si existen ambos grupos
    if mask_m.any() and mask_f.any():
        # Usar máscaras para indexar
        mu_m = X[mask_m].mean(axis=0)  # embedding medio de hombres
        mu_f = X[mask_f].mean(axis=0)  # embedding medio de mujeres
        # Dirección = hombres - mujeres
        out["gender_M_minus_F"] = _unit(mu_m - mu_f)

    # ---------- DIRECCIÓN EDAD ----------
    # Pasamos AGE a numérico
    age = pd.to_numeric(df_meta.get("AGE"), errors="coerce")

    # Solo calculamos si tenemos al menos 8 hablantes con edad conocida
    if age.notna().sum() >= 8:
        # Percentiles 25 (jóvenes) y 75 (mayores)
        q1, q4 = np.nanpercentile(age, 25), np.nanpercentile(age, 75)

        # Máscaras: jóvenes <= Q1, mayores >= Q4
        mask_y = age <= q1
        mask_o = age >= q4

        if mask_y.any() and mask_o.any():
            mu_o = X[mask_o].mean(axis=0)  # embedding medio de mayores
            mu_y = X[mask_y].mean(axis=0)  # embedding medio de jóvenes
            out["age_old_minus_young"] = _unit(mu_o - mu_y)

    return out

def build_corr_based_direction_in_Z(corr_df: pd.DataFrame, k_total: int, topk: int) -> np.ndarray:
    """
        Construye una dirección semántica en el espacio Z (el espacio reducido por PCA, donde cada coordenada es un z_i)
        a partir de la relación estadística (correlación de Pearson) entre las componentes PCA y un atributo (como edad o género)

        Una dirección semántica significa: un vector en ese espacio que apunta hacia un “cambio” interpretable.
        Por ejemplo:
            - Avanzar en dirección positiva podría significar “hacer la voz más masculina”.
            - Avanzar en dirección negativa podría significar “hacer la voz más femenina”

        Pasos:
          1) Ordena las componentes z_i por correlación de Pearson (|r|).
                - Cada eje z1, z2,... del PCA puede estar más o menos correlacionado con “ser hombre/mujer” o con “ser joven/viejo”.
                - La correlación de Pearson r mide cuán linealmente relacionada está esa variable con el atributo
                  (1 = relación positiva perfecta, -1 = relación negativa perfecta, 0 = ninguna relación).

          2) Selecciona las top-k más relevantes.
                - Ejemplo: si z3 y z7 son los que más se correlacionan con el género, usamos solo esos dos.

          3) Asigna pesos proporcionales a |r| y conserva el signo (+/-).
                - Si z3 tiene correlación r = 0.8 y z7 tiene r = -0.4, entonces z3 es el doble de importante.
                - Conservamos el signo: positivo significa que valores grandes de ese eje se asocian con “más masculino”
                (o “más viejo”), negativo significa lo contrario.

          4) Combina en un vector v de dimensión k_total.
                - Este vector tiene tantas entradas como ejes (k_total = nº de componentes PCA).
                - Solo en los índices seleccionados (ej: z3 y z7) ponemos valores distintos de 0, proporcionales a esas correlaciones.

          5) Normaliza v → dirección unitaria.

        Interpretación:
          - Género (M=1, F=0): signo + → más masculino, signo - → más femenino.
          - Edad: signo + → mayor edad, signo - → más joven.

        Devuelve un vector unitario que sirve como “slider semántico” en Z.
    """

    # Seleccionamos las top-k filas (por |r|) y trabajamos sobre una copia
    use = corr_df.head(max(1, topk)).copy()

    # Pesos proporcionales a |r|. Si todas las |r|≈0, usamos 1.0 para evitar división por cero.
    use["w"] = use["pearson_r"].abs()
    if use["w"].sum() <= 1e-9:
        use["w"] = 1.0
    use["w"] = use["w"] / use["w"].sum()

    # Vector de dirección en Z: inicialmente todo ceros (longitud = nº total de componentes PCA)
    v = np.zeros((k_total,), dtype=np.float32)

    # Para cada componente seleccionada, añadimos su contribución con signo
    for _, row in use.iterrows():
        comp = str(row["component"])  # p.ej. "z3"
        r = float(row["pearson_r"])  # correlación de esa z con el atributo
        idx = int(comp.replace("z", "")) - 1  # "z3" -> índice 2 (0‑based)
        sign = 1.0 if r >= 0 else -1.0  # conserva el sentido: + si r>0, - si r<0
        v[idx] += sign * float(row["w"])  # acumula peso (proporcional a |r|) en la posición de esa z

    # Normalizamos el vector final para que sea una dirección unitaria
    return _unit(v)


def save_Z_coordinates(Z: np.ndarray, df_src: pd.DataFrame, path: Path):
    """
        Guarda las coordenadas Z

        Resultado:
        Un archivo CSV con las coordenadas en Z + columnas de identificación
        que luego sirve para analizar, correlacionar o incluso manipular voces.
    """
    cols = [f"z{i+1}" for i in range(Z.shape[1])]
    zdf = pd.DataFrame(Z, columns=cols)
    for id_col in ["corpus", "speaker_id"]:
        if id_col in df_src.columns:
            zdf.insert(0, id_col, df_src[id_col].values)
    zdf.to_csv(path, index=False)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    # 1) Cargar embeddings
    # Cargamos el CSV unificado (todas las voces de todos los corpus)
    emb_df = load_merged_embeddings()
    # Normalizamos los IDs de speaker según el corpus (para que no haya inconsistencias)
    emb_df = normalize_ids(emb_df)
    # Guardamos una copia para referencia y extraemos las columnas de embeddings (e0...eN)
    emb_df.to_csv(OUT_DIR / "merged_per_speaker_copy.csv", index=False)
    emb_cols = _emb_cols(emb_df)
    if not emb_cols:
        raise ValueError("El CSV no contiene columnas de embedding e0..eN.")
    X = emb_df[emb_cols].to_numpy(dtype=np.float32)

    # 2) Anexar metadatos
    # -------------------
    # Añadimos al DataFrame información adicional de los locutores:
    # - Edad
    # - Género
    # - Acento
    # - Región
    # Esto se hace leyendo ficheros auxiliares de cada corpus
    df_all = attach_metadata(emb_df)
    df_all.to_csv(OUT_DIR / "embeddings_with_metadata.csv", index=False)

    # 3) PCA
    PCA_DIR = Path("./pca_xtts_k64")
    scaler_path = PCA_DIR / "scaler.joblib"
    pca_path = PCA_DIR / "pca.joblib"

    if scaler_path.exists() and pca_path.exists():
        scaler = load(scaler_path)
        pca = load(pca_path)

        # Normalizamos y proyectamos embeddings → Z
        Xs = scaler.transform(X)
        Z = pca.transform(Xs)
    else:
        # Si no hay PCA entrenado, no se puede continuar.
        raise FileNotFoundError(
            f"No se encontraron '{scaler_path}' o '{pca_path}'. "
            f"Coloca tus modelos preentrenados o ajusta PCA_DIR."
        )

    # Guardar coordenadas Z
    save_Z_coordinates(Z, df_all, OUT_DIR / "Z_coordinates.csv")

    # 4) Correlaciones
    # Calculamos qué ejes en Z se correlacionan más con la edad y el género.
    age_corr_df, gender_corr_df = correlations(Z, df_all)
    age_corr_df.to_csv(OUT_DIR / "correlaciones_Z_con_edad.csv", index=False)
    gender_corr_df.to_csv(OUT_DIR / "correlaciones_Z_con_genero.csv", index=False)

    # Creamos direcciones semánticas en Z a partir de las correlaciones top-k
    k_total = Z.shape[1]
    gender_corr_dir_z = build_corr_based_direction_in_Z(gender_corr_df, k_total, TOPK_FOR_Z_DIRS)
    age_corr_dir_z    = build_corr_based_direction_in_Z(age_corr_df,    k_total, TOPK_FOR_Z_DIRS)

    # Guardamos esas direcciones
    np.savez(OUT_DIR / "semantic_z_from_corr.npz",
             gender_corr_dir_z=gender_corr_dir_z,
             age_corr_dir_z=age_corr_dir_z)

    # Guardamos también un JSON que documenta qué significa cada dirección
    info = {
        "k_total": int(k_total),
        "topk_used_for_Z_dirs": int(TOPK_FOR_Z_DIRS),
        "gender_dir_z": {
            "points_to": "male (M=1)",
            "source": "correlations top-k by |r|",
            "npz_key": "gender_corr_dir_z",
            "top_component": str(gender_corr_df.iloc[0]["component"]) if not gender_corr_df.empty else None,
            "top_r": float(gender_corr_df.iloc[0]["pearson_r"]) if not gender_corr_df.empty else None
        },
        "age_dir_z": {
            "points_to": "older (higher AGE)",
            "source": "correlations top-k by |r|",
            "npz_key": "age_corr_dir_z",
            "top_component": str(age_corr_df.iloc[0]["component"]) if not age_corr_df.empty else None,
            "top_r": float(age_corr_df.iloc[0]["pearson_r"]) if not age_corr_df.empty else None
        }
    }
    with (OUT_DIR / "semantic_axes_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    # 5) Direcciones semánticas en espacio original (512D)
    # ----------------------------------------------------
    # También calculamos direcciones en el espacio nativo de embeddings:
    # - Diferencia media entre hombres y mujeres.
    # - Diferencia media entre jóvenes y mayores.
    # Estas sirven como alternativa directa a las obtenidas en Z.
    dirs = semantic_directions(df_all, X)
    if dirs:
        np.savez(OUT_DIR / "semantic_directions.npz", **dirs)
        print("[OK] Direcciones semánticas (512D):", list(dirs.keys()))
    else:
        print("[WARN] No se pudieron calcular direcciones semánticas (¿faltan metadatos?).")

    # Resumen de archivos generados
    print("[OK] Z_coordinates.csv:", OUT_DIR / "Z_coordinates.csv")
    print("[OK] correlaciones_Z_con_genero.csv:", OUT_DIR / "correlaciones_Z_con_genero.csv")
    print("[OK] correlaciones_Z_con_edad.csv:", OUT_DIR / "correlaciones_Z_con_edad.csv")
    print("[OK] semantic_z_from_corr.npz:", OUT_DIR / "semantic_z_from_corr.npz")
    print("[OK] semantic_axes_info.json:", OUT_DIR / "semantic_axes_info.json")
    print("[OK] embeddings_with_metadata.csv:", OUT_DIR / "embeddings_with_metadata.csv")
    print("[OK] semantic_directions.npz (512D):", OUT_DIR / "semantic_directions.npz")

if __name__ == "__main__":
    main()
