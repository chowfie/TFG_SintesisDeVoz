from joblib import load as joblib_load
from TTS.api import TTS
from pathlib import Path
from datetime import datetime
import sys
import traceback
import numpy as np
import pandas as pd
import torch
from PySide6.QtWidgets import QStyleFactory
from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QSlider,
    QHBoxLayout, QVBoxLayout, QPushButton, QComboBox, QMessageBox,
    QFrame, QTextEdit, QSizePolicy
)
from PySide6.QtGui import QFont, QPalette, QColor, QPixmap
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput


# ------------------------------------------------------------
# Rutas y ajustes
# ------------------------------------------------------------
LATENT_DIR = Path("./latent_out")
EMB_CSV   = LATENT_DIR / "embeddings_with_metadata.csv"
PCA_DIR = Path("./pca_xtts_k64")
SCALER_JOBLIB = PCA_DIR / "scaler.joblib"
PCA_JOBLIB = PCA_DIR / "pca.joblib"
DIRS_NPZ  = LATENT_DIR / "semantic_directions.npz"
DIRS_Z_NPZ = LATENT_DIR / "semantic_z_from_corr.npz"
DIRS_REG_NPZ = LATENT_DIR / "semantic_directions_regression.npz"

LOGO_PATH = Path("logo.png")

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

def safe_mean(X: np.ndarray, mask) -> np.ndarray:
    """
        Media robusta con máscara booleana.
        Si la máscara no tiene ningún True, devuelve media global.
        Si la máscara viene rara (None, dtype no bool, shape distinta), también cae a global.
    """
    X = np.asarray(X, dtype=np.float32)
    if not isinstance(mask, (np.ndarray, pd.Series)) or mask is None:
        return X.mean(axis=0).astype(np.float32)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] != X.shape[0] or mask.sum() == 0:
        return X.mean(axis=0).astype(np.float32)
    return X[mask].mean(axis=0).astype(np.float32)

# ------------------------------------------------------------
# Artifacts
# ------------------------------------------------------------
def load_artifacts_512():
    """
        Carga todos los datos necesarios para trabajar en el espacio original de embeddings (512D).
        - Abre el CSV con embeddings de cada locutor y sus metadatos.
        - Calcula un "embedding medio" para hombres, mujeres y todos juntos.
        - Carga las direcciones semánticas (vectores) que representan:
            * diferencia hombre ↔ mujer
            * diferencia mayor ↔ joven
            * diferencia agudo ↔ grave en pitch
        - Devuelve junto en un diccionario listo para usar.
    """
    if not EMB_CSV.exists():
        raise FileNotFoundError(
            f"No encuentro {EMB_CSV}. "
            f"Ejecuta learn_axes.py para generar 'embeddings_with_metadata.csv'."
        )
    df = pd.read_csv(EMB_CSV, low_memory=False)

    # Detectar columnas de embeddings (e0...e511)
    emb_cols = [c for c in df.columns if c.startswith("e")]
    if not emb_cols:
        raise RuntimeError("El CSV no tiene columnas e0..eN (embeddings 512D).")

    X = df[emb_cols].to_numpy(dtype=np.float32)

    # Normalizar la columna GENDER a M/F
    g = df.get("GENDER")
    if g is None:
        g = pd.Series([None] * len(df))
    g = (
        g.astype(str)
         .str.upper()
         .replace({"MALE": "M", "FEMALE": "F", "HOMBRE": "M", "MUJER": "F"})
    )
    mask_M = (g == "M")
    mask_F = (g == "F")

    # Calculamos promedios (vectores medios) para cada grupo
    mu_M = safe_mean(X, mask_M)
    mu_F = safe_mean(X, mask_F)
    mu_all = X.mean(axis=0).astype(np.float32)

    # Cargar direcciones semánticas ya calculadas (género/edad/pitch) en 512D
    if not DIRS_NPZ.exists():
        raise FileNotFoundError(
            f"No encuentro {DIRS_NPZ}. "
            f"Ejecuta learn_axes.py para generar 'semantic_directions.npz'."
        )
    npz = np.load(DIRS_NPZ, allow_pickle=True)
    dir_gender = npz["gender_M_minus_F"] if "gender_M_minus_F" in npz.files else None
    dir_age    = npz["age_old_minus_young"] if "age_old_minus_young" in npz.files else None

    if dir_gender is None or dir_age is None:
        raise RuntimeError(
            "semantic_directions.npz no contiene 'gender_M_minus_F' o 'age_old_minus_young'. "
            "Revisa learn_axes.py / metadatos."
        )

    # Eje Pitch
    dir_pitch = None
    if "pitch_high_minus_low" in npz.files:
        dir_pitch = npz["pitch_high_minus_low"]

    # Normalizar todos los vectores a norma 1
    dir_gender = unit(dir_gender)
    dir_age    = unit(dir_age)
    dir_pitch  = unit(dir_pitch) if dir_pitch is not None else None

    return {
        "X": X,
        "mu_M": unit(mu_M),
        "mu_F": unit(mu_F),
        "mu_all": unit(mu_all),
        "dir_gender": dir_gender,
        "dir_age": dir_age,
        "dir_pitch": dir_pitch,   # puede ser None si no existe
    }

def load_artifacts_Z():
    """
       Carga los datos necesarios para trabajar en el espacio reducido por PCA (espacio Z).

       - Abre el scaler y el PCA entrenados previamente (para transformar de 512D a Z y volver).
       - Carga las direcciones semánticas en Z (vectores que apuntan a hombre↔mujer, viejo↔joven y grave↔agudo).
       - Normaliza esas direcciones para que tengan norma 1.
       - Devuelve en un diccionario listo para usar.
       """
    # Verifica que existan scaler y PCA entrenados
    if not (SCALER_JOBLIB.exists() and PCA_JOBLIB.exists()):
        raise FileNotFoundError("Falta PCA: coloca scaler.joblib y pca.joblib en ./pca_xtts_k64")
    scaler = joblib_load(SCALER_JOBLIB)
    pca = joblib_load(PCA_JOBLIB)

    # Verifica que exista el archivo con las direcciones semánticas en Z
    if not DIRS_Z_NPZ.exists():
        raise FileNotFoundError("No encuentro semantic_z_from_corr.npz en latent_out")
    npz = np.load(DIRS_Z_NPZ, allow_pickle=True)

    return {
        "scaler": scaler,
        "pca": pca,
        "dir_gender_z": unit(npz["gender_corr_dir_z"]) if "gender_corr_dir_z" in npz.files else None,
        "dir_age_z":    unit(npz["age_corr_dir_z"])    if "age_corr_dir_z"    in npz.files else None,
        "dir_pitch_z":  unit(npz["pitch_corr_dir_z"])  if "pitch_corr_dir_z"  in npz.files else None,
    }

def load_artifacts_regression():
    """
        Carga las direcciones aprendidas por regresión en 512D (independiente del PCA).
        - Lee 'semantic_directions_regression.npz' con:
            * gender_reg_512
            * age_reg_512
            * pitch_reg_512 (si existe)
        - Reutiliza los embeddings medios (mu_M, mu_F, mu_all) calculados en 512D
          para poder usar choose_base_embedding con el mismo criterio.
        - Devuelve un diccionario con las claves esperadas por edit_embedding_512.
    """
    # Reutilizamos medios y X desde el loader 512D (no altera nada)
    art_base = load_artifacts_512()

    if not DIRS_REG_NPZ.exists():
        raise FileNotFoundError(
            f"No encuentro {DIRS_REG_NPZ}. Ejecuta learn_axes_regression.py para generar este archivo."
        )

    npz = np.load(DIRS_REG_NPZ, allow_pickle=True)

    dir_gender = npz["gender_reg_512"] if "gender_reg_512" in npz.files else None
    dir_age    = npz["age_reg_512"]    if "age_reg_512"    in npz.files else None
    dir_pitch  = npz["pitch_reg_512"]  if "pitch_reg_512"  in npz.files else None

    if dir_gender is None or dir_age is None or dir_pitch is None:
        raise RuntimeError(
            "semantic_directions_regression.npz no contiene 'gender_reg_512' o 'age_reg_512' o 'pitch_reg_512'."
        )

    # Normalizamos las direcciones a norma 1 (consistente con tu pipeline)
    dir_gender = unit(dir_gender)
    dir_age    = unit(dir_age)
    dir_pitch  = unit(dir_pitch) if dir_pitch is not None else None

    return {
        "mu_M": art_base["mu_M"],
        "mu_F": art_base["mu_F"],
        "mu_all": art_base["mu_all"],
        "dir_gender": dir_gender,
        "dir_age": dir_age,
        "dir_pitch": dir_pitch,   # puede ser None si ese eje no está en el NPZ
    }

# ------------------------------------------------------------
# Edit Embeddings
# ------------------------------------------------------------
def edit_embedding_via_Z(e_base_512: np.ndarray, z_dirs: dict,
                         alpha_gender: float, alpha_age: float, alpha_pitch: float) -> np.ndarray:
    """
        Edita un embedding en el espacio Z (PCA).

        - Toma un embedding base (512D).
        - Lo proyecta al espacio Z con el scaler y el PCA.
        - Aplica cambios en género, edad y pitch usando las direcciones de Z.
        - Reconstruye el embedding editado de vuelta a 512D.
        - Devuelve el embedding final, normalizado.
    """
    # Recuperamos scaler y PCA del diccionario
    scaler = z_dirs["scaler"]; pca = z_dirs["pca"]

    # Recuperamos direcciones semánticas en Z
    dg = z_dirs["dir_gender_z"]
    da = z_dirs["dir_age_z"]
    dp = z_dirs.get("dir_pitch_z", None)

    # Pasamos el embedding base de 512D a Z
    z = pca.transform(scaler.transform(e_base_512.reshape(1, -1))).reshape(-1)

    # Copiamos del embedding en Z para modificarlo
    z_edit = z.copy()
    # Aplicamos desplazamientos en género, edad y pitch
    if dg is not None and alpha_gender != 0.0:
        z_edit += float(alpha_gender) * dg
    if da is not None and alpha_age    != 0.0:
        z_edit += float(alpha_age)    * da
    if dp is not None and alpha_pitch  != 0.0:
        z_edit += float(alpha_pitch)  * dp

    # Reconstruimos:Z
    Xs_rec = pca.inverse_transform(z_edit.reshape(1, -1))
    e_rec = scaler.inverse_transform(Xs_rec).reshape(-1).astype(np.float32)

    # Normalizamos para que sea un embedding válido
    return unit(e_rec)

def edit_embedding_512(e_base: np.ndarray,
                                  dir_gender: np.ndarray, dir_age: np.ndarray, dir_pitch: np.ndarray | None,
                                  alpha_gender: float, alpha_age: float, alpha_pitch: float) -> np.ndarray:
    """
        Modifica un embedding en 512D aplicando género, edad y pitch.

        - Empieza desde un embedding base (hombre, mujer o neutro).
        - Suma el efecto del slider de género
        - Suma el efecto del slider de edad
        - Suma el efecto del slider pitch
        - Devuelve el nuevo embedding normalizado (para que no “explote” la escala).
    """

    # Aplicar género y edad
    e_edit = e_base + float(alpha_gender) * dir_gender + float(alpha_age) * dir_age + float(alpha_pitch) * dir_pitch

    # Normalizar el vector resultante
    return unit(e_edit)

def choose_base_embedding(art, gender_value: float, thr: float = 0.6) -> np.ndarray:
    """
        Elige un embedding base (512D) según el valor del slider de género.

        - Si el slider está muy hacia masculino (>= thr) → devuelve el embedding medio de hombres.
        - Si está muy hacia femenino (<= -thr) → devuelve el embedding medio de mujeres.
        - Si está en una zona intermedia → devuelve un embedding neutro (media de todos).
    """

    if gender_value >= thr:
        # Usar embedding medio de hombres
        return art["mu_M"].copy()
    if gender_value <= -thr:
        # Usar embedding medio de mujeres
        return art["mu_F"].copy()
    # Si no está ni en masculino ni femenino fuerte, usar media global
    return art["mu_all"].copy()

# ------------------------------------------------------------
# XTTS
# ------------------------------------------------------------
def prepare_xtts(tts: TTS):
    """
        Prepara el modelo XTTS para usar embeddings personalizados.

        - Busca en el modelo las "huellas" de voz ya precargadas.
        - Calcula un promedio de esas huellas → nos da un punto de partida neutro.
        - Devuelve:
            * el modelo XTTS ya cargado
            * el gestor de locutores (speaker_manager)
            * el dispositivo donde corre (CPU/GPU)
            * el embedding medio de referencia (gpt_mean)
        Si no encuentra huellas, avisa que hace falta cargar un locutor de referencia o un WAV.
    """
    model = tts.synthesizer.tts_model
    device = next(model.parameters()).device
    # Obtenemos el speaker_manager
    spkman = getattr(model, "speaker_manager", None)
    if spkman is None:
        raise RuntimeError("El modelo XTTS no tiene speaker_manager")

    latents = []
    # Comprobamos que dentro de speaker_manager hay locutores cargados (diccionario)
    if hasattr(spkman, "speakers") and isinstance(spkman.speakers, dict):
        for _, entry in spkman.speakers.items():
            # Si es un diccionario y tiene la clave "gpt_cond_latent" (la huella de voz)
            if isinstance(entry, dict) and "gpt_cond_latent" in entry:
                # Confirmamos que es un tensor de PyTorch (embedding válido)
                t = entry["gpt_cond_latent"]
                if isinstance(t, torch.Tensor):
                    # Lo movemos al dispositivo correcto (CPU o GPU) y lo guardamos
                    latents.append(t.to(device))

    # Si no hemos encontrado ninguna huella, no podemos continuar
    if not latents:
        raise RuntimeError(
            "No hay gpt_cond_latent precargado. "
            "Carga algún speaker de referencia o usa get_conditioning_latents con un WAV."
        )
    # Calculamos la media de todas las huellas de voz encontradas: voz base neutra
    gpt_mean = torch.mean(torch.stack(latents, dim=0), dim=0)
    # Devolvemos necesario para usar XTTS
    return model, spkman, device, gpt_mean


# ------------------------------------------------------------
# Ejecucion
# ------------------------------------------------------------
class SynthesisWorker(QThread):
    done   = Signal(str)  # ruta del wav generado
    failed = Signal(str)  # errores
    status = Signal(str)  # mensajes de estado

    def __init__(self, language: str, text: str, gender: float, age: float, out_dir: Path,
                 pitch: float = 0.0, mode: str = "semantic_512d"):
        super().__init__()
        self.language = language
        self.text = text
        self.gender = float(gender)
        self.age = float(age)
        self.pitch = float(pitch)
        self.mode = mode
        self.out_dir = out_dir

    def run(self):
        try:
            self.status.emit("# ----------------------------------------------------------------")
            self.status.emit("Cargando artefactos (embeddings/direcciones)...")
            art = load_artifacts_512()

            # 1) Seleccionamos un embedding base (512D) según el slider de género
            e_base = choose_base_embedding(art, self.gender, thr=0.6)

            # 2) Modificamos el embedding según el modo seleccionado
            if self.mode == "semantic_512d":
                # Usamos direcciones semánticas en el espacio original (512D)
                self.status.emit("Calculando valores con direcciones semanticas...")
                e_edit = edit_embedding_512(
                    e_base,
                    art["dir_gender"],
                    art["dir_age"],
                    art.get("dir_pitch", None),
                    alpha_gender=self.gender,
                    alpha_age=self.age,
                    alpha_pitch=self.pitch
                )
            elif self.mode == "correlations_z":
                self.status.emit("Calculando valores con espacio reducido Z y correlaciones...")
                self.status.emit("Cargando PCA y direcciones en Z...")
                z_dirs = load_artifacts_Z()
                e_edit = edit_embedding_via_Z(
                    e_base_512=e_base,
                    z_dirs=z_dirs,
                    alpha_gender=self.gender,
                    alpha_age=self.age,
                    alpha_pitch=self.pitch
                )

            elif self.mode == "regression_512d":
                self.status.emit("Calculando valores con regresión 512D...")
                reg = load_artifacts_regression()
                e_edit = edit_embedding_512(
                    e_base,
                    reg["dir_gender"],
                    reg["dir_age"],
                    reg.get("dir_pitch", None),
                    alpha_gender=self.gender,
                    alpha_age=self.age,
                    alpha_pitch=self.pitch
                )

            else:
                raise ValueError(f"Modo desconocido: {self.mode}")

            # 3) Cargamos el modelo XTTS v2 listo para sintetizar
            self.status.emit("Cargando modelo XTTS v2...")
            tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                           progress_bar=False,
                           gpu=torch.cuda.is_available())

            # 4) Preparamos XTTS para aceptar nuestro embedding modificado
            model, spkman, device, gpt_mean = prepare_xtts(tts)
            spk_id = "edited_voice_gui"

            # Registramos nuestro embedding modificado en el speaker_manager
            spkman.speakers[spk_id] = {
                "gpt_cond_latent": gpt_mean,
                "speaker_embedding": torch.tensor(e_edit, dtype=torch.float32, device=device).view(1, -1, 1)
            }

            # 5) Generamos archivo WAV de salida con timestamp
            self.out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_wav = self.out_dir / f"xtts_edit_{ts}.wav"

            self.status.emit("Sintetizando...")
            tts.tts_to_file(text=self.text, language=self.language, speaker=spk_id, file_path=str(out_wav))
            self.done.emit(str(out_wav))

        except Exception as e:
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")

# ------------------------------------------------------------
# Interfaz gráfica
# ------------------------------------------------------------
def apply_pretty_theme(app):
    """
        Aplica un tema visual bonito a la aplicación Qt.

        - Cambia el estilo base de la interfaz (tipografía, colores, fondo).
        - Define una paleta de colores personalizada (azul oscuro + acentos verdes y azules).
        - Aplica reglas de estilo (QSS) para que los widgets se vean más modernos:
          * Botones con degradados y hover
          * Sliders con pista redondeada
          * Cuadros de texto y combos con bordes suaves
    """
    # 1) Estilo base
    app.setStyle(QStyleFactory.create("Fusion"))

    # 2) Tipografía
    app.setFont(QFont("Segoe UI", 10))

    # 3) Palette verdiazul
    base = QColor("#0e1726")
    panel = QColor("#162135")
    text = QColor("#e6eef8")
    accent2 = QColor("#4dabf7")
    disabled = QColor("#8391a7")

    # 4) Creamos una paleta y asignamos colores a elementos básicos
    pal = QPalette()
    pal.setColor(QPalette.Window, base)
    pal.setColor(QPalette.WindowText, text)
    pal.setColor(QPalette.Base, panel)
    pal.setColor(QPalette.AlternateBase, base)
    pal.setColor(QPalette.Text, text)
    pal.setColor(QPalette.Button, panel)
    pal.setColor(QPalette.ButtonText, text)
    pal.setColor(QPalette.Highlight, accent2)
    pal.setColor(QPalette.HighlightedText, QColor("#0b1020"))
    pal.setColor(QPalette.ToolTipBase, panel)
    pal.setColor(QPalette.ToolTipText, text)
    pal.setColor(QPalette.Disabled, QPalette.Text, disabled)
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, disabled)
    app.setPalette(pal)

    # 5) QSS: estilos avanzados (CSS para Qt)
    app.setStyleSheet("""
        QWidget {
            background: #0e1726;
            color: #e6eef8;
        }
        QLineEdit, QComboBox, QTextEdit {
            background: #162135;
            border: 1px solid #24324a;
            border-radius: 12px;
            padding: 8px 10px;
            selection-background-color: #4dabf7;
            selection-color: #0b1020;
        }
        QLineEdit:focus, QComboBox:focus, QTextEdit:focus {
            border: 1px solid #20c997;
        }
        QPushButton {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #20c997, stop:1 #4dabf7);
            border: none;
            color: white;
            padding: 10px 14px;
            border-radius: 14px;
            font-weight: 600;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #2fdfb0, stop:1 #67b7fb);
        }
        QPushButton:pressed {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #1aa982, stop:1 #3b8fd2);
        }
        QPushButton:disabled {
            background: #24324a;
            color: #98a6bd;
        }
        QLabel {
            color: #cfe4ff;
        }
        QStatusBar, .QFrame {
            background: transparent;
            color: #9fb3cc;
        }
        QSlider::groove:horizontal {
            border: 1px solid #24324a;
            height: 8px;
            background: #1b2942;
            border-radius: 6px;
            margin: 0 10px;
        }
        QSlider::handle:horizontal {
            background: #20c997;
            border: 2px solid #0e1726;
            width: 20px;
            height: 20px;
            margin: -7px -10px;
            border-radius: 10px;
        }
        QSlider::handle:horizontal:hover {
            background: #4dabf7;
        }
        QComboBox QAbstractItemView {
            background: #162135;
            color: #e6eef8;
            selection-background-color: #4dabf7;
            selection-color: #0b1020;
            border: 1px solid #24324a;
            border-radius: 10px;
        }
    """)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # ------------------------------------------------------------
        # Configuración inicial de la ventana principal
        # ------------------------------------------------------------
        self.setWindowTitle("CREAUDIO")
        self.resize(900, 540)

        # ---- Estado de audio y reproductor integrado ----
        # Guarda la ruta del último WAV generado
        # y configura el reproductor de audio de Qt
        self.last_wav = None
        self.player = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_out)

        # Cabecera
        header = QHBoxLayout()
        header.addStretch()  # centra el logo
        self.logo_lbl = QLabel()
        self.logo_lbl.setObjectName("logoLabel")
        # Carga logo
        self._load_logo(
            path=LOGO_PATH,
            target_h=150  # alto del logo
        )
        header.addWidget(self.logo_lbl)
        header.addStretch()

        # Espacio debajo del logo para separarlo del input de texto
        header_wrap = QVBoxLayout()
        header_wrap.addLayout(header)
        header_wrap.addSpacing(14)

        # FILA 1: Texto a sintetizar
        self.text_label = QLabel("Texto:")
        self._make_bold(self.text_label, pts=11)  # etiqueta en negrita y un poco más grande
        self.text_edit = QLineEdit("Esto es una prueba de creación de voz personalizada") # Texto por defecto
        row_text = QHBoxLayout()
        row_text.addWidget(self.text_label)
        row_text.addSpacing(10)
        row_text.addWidget(self.text_edit, 1)

        # FILA 2: Idioma + Modo de edición de embedding
        # Idioma
        self.lang_label = QLabel("Idioma:")
        self._make_bold(self.lang_label, pts=11)
        self.lang_combo = QComboBox()
        # Idiomas más estables en XTTS
        for code in ["es","en","fr","de","it","pt","pl","tr","ru","zh-cn","ja","ko"]:
            self.lang_combo.addItem(code, code)

        # Modo
        self.mode_label = QLabel("Modo:")
        self._make_bold(self.mode_label, pts=11)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Semánticas 512D", "semantic_512d")
        self.mode_combo.addItem("Correlaciones en Z", "correlations_z")
        self.mode_combo.addItem("Regresión 512D", "regression_512d")

        row_top = QHBoxLayout()
        # alineados a la izquierda
        row_top.addWidget(self.lang_label)
        row_top.addSpacing(8)
        row_top.addWidget(self.lang_combo)
        row_top.addSpacing(24)
        row_top.addWidget(self.mode_label)
        row_top.addSpacing(8)
        row_top.addWidget(self.mode_combo)
        row_top.addStretch()

        # Separador con título: "Configuración de la voz"
        title_cfg = QLabel("Configuración de la voz")
        title_cfg.setAlignment(Qt.AlignCenter)
        title_cfg.setStyleSheet("font-size: 16px; font-weight: 600; color: #cfe4ff;")
        line_top = QFrame()
        line_top.setFrameShape(QFrame.HLine)
        line_top.setFrameShadow(QFrame.Sunken)
        line_bot = QFrame()
        line_bot.setFrameShape(QFrame.HLine)
        line_bot.setFrameShadow(QFrame.Sunken)

        # ------------------------------------------------------------
        # Sliders de control de parámetros de voz
        #   - Género:   -2 = Femenino, +2 = Masculino
        #   - Edad:     -2 = Joven, +2 = Mayor
        #   - Tono:     -2 = Grave, +2 = Agudo
        # ------------------------------------------------------------
        self.gender_section = self._make_slider_section(
            title="Género", left_text="Femenino", right_text="Masculino",
            vmin=-3.0, vmax=3.0, v0=0.0, step=0.1
        )
        self.age_section = self._make_slider_section(
            title="Edad", left_text="Joven", right_text="Mayor",
            vmin=-3.0, vmax=3.0, v0=0.0, step=0.1
        )
        self.pitch_section = self._make_slider_section(
            title="Tono", left_text="Grave", right_text="Agudo",
            vmin=-3.0, vmax=3.0, v0=0.0, step=0.1
        )

        v_sliders = QVBoxLayout()
        v_sliders.addWidget(self.gender_section["widget"])
        v_sliders.addWidget(self.age_section["widget"])
        v_sliders.addWidget(self.pitch_section["widget"])

        # Cajita con valor de slider
        self.gender_box = self.gender_section["box"]
        self.age_box    = self.age_section["box"]
        self.pitch_box  = self.pitch_section["box"]

        # ------------------------------------------------------------
        # Botones de acción principales
        #   - Sintetizar: genera un nuevo audio
        #   - Reproducir: reproduce el último audio generado
        # ------------------------------------------------------------
        # Botón Sintetizar
        self.btn_synth = QPushButton("Sintetizar")
        self._make_bold(self.btn_synth, pts=11)
        self.btn_synth.clicked.connect(self.on_synthesize)
        # Botón Reproducir
        self.btn_play  = QPushButton("Reproducir")
        self._make_bold(self.btn_play, pts=11)
        # Deshabilitado porque no hay audio al inicio
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self.on_play)

        row_btns = QHBoxLayout()
        row_btns.addStretch()
        row_btns.addWidget(self.btn_synth)
        row_btns.addSpacing(12)
        row_btns.addWidget(self.btn_play)
        row_btns.addStretch()

        # Área de logs
        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setPlaceholderText("Logs de síntesis...")

        # ------------------------------------------------------------
        # Layout raíz de la ventana
        # ------------------------------------------------------------
        root = QWidget()
        lay = QVBoxLayout(root)
        lay.addLayout(header_wrap) # cabecera con logo y espacio inferior
        lay.addLayout(row_text)    # texto
        lay.addLayout(row_top)     # idioma + modo
        lay.addWidget(line_top)
        lay.addWidget(title_cfg)   # título sección sliders
        lay.addWidget(line_bot)
        lay.addLayout(v_sliders)   # sliders
        lay.addLayout(row_btns)    # botones
        lay.addWidget(self.logs)   # logs
        self.setCentralWidget(root)

    # ------------------------------------------------------------
    # Estilos titulos
    # ------------------------------------------------------------
    def _make_bold(self, widget: QWidget, pts: int = 11):
        """Aplica negrita y tamaño de fuente ligeramente mayor a un widget."""
        f = widget.font()
        f.setBold(True)
        base_size = f.pointSize() if f.pointSize() > 0 else 10
        f.setPointSize(max(base_size, pts))
        widget.setFont(f)

    # ------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------
    def _append_log(self, text: str):
        """
            Añade una línea al área de logs, acumulando el contenido existente.
            Si el widget de logs no existe aún, se ignora silenciosamente.
        """
        try:
            self.logs.append(text)
        except AttributeError:
            # Por si se llama antes de crear self.logs
            pass

    # ------------------------------------------------------------
    # Construcción de secciones de sliders
    # ------------------------------------------------------------
    def _make_slider_section(self, title, left_text, right_text, vmin, vmax, v0, step=0.1):
        """
            Crea una sección completa:
                [Título en negrita]
                valor_izquierda  [slider horizontal]  valor_derecha  [cajita]

            Devuelve dict con:
              - 'widget': QWidget con la sección completa
              - 'slider': QSlider configurado
              - 'box': QLineEdit numérica vinculada al slider
              - 'scale': factor de escala (100) para mapear floats a enteros
        """
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Título
        lab_title = QLabel(f"{title}:")
        self._make_bold(lab_title, pts=12)
        layout.addWidget(lab_title)

        # Fila del control
        scale = 100
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(vmin * scale))
        slider.setMaximum(int(vmax * scale))
        slider.setValue(int(v0 * scale))
        slider.setSingleStep(int(step * scale))

        slider.setFixedWidth(800)

        box = QLineEdit(f"{v0:.2f}")
        box.setFixedWidth(70)

        lab_left = QLabel(left_text)
        lab_right = QLabel(right_text)

        row = QHBoxLayout()
        row.addWidget(lab_left)
        row.addSpacing(8)
        row.addWidget(slider, 1)
        row.addSpacing(8)
        row.addWidget(lab_right)
        row.addSpacing(8)
        row.addWidget(box)
        layout.addLayout(row)

        # Sincronización slider y cajita
        def on_slide(x):
            box.setText(f"{x/scale:.2f}")

        def on_edit():
            try:
                v = float(box.text())
                v = max(vmin, min(vmax, v))
                slider.setValue(int(v * scale))
            except Exception:
                pass

        slider.valueChanged.connect(on_slide)
        box.editingFinished.connect(on_edit)

        return {"widget": container, "slider": slider, "box": box, "scale": scale}

    # ------------------------------------------------------------
    # Logo
    # ------------------------------------------------------------
    def _load_logo(self, path: Path, target_h: int = 150):
        """
            Carga el logo y lo escala
        """
        try:
            pix = QPixmap(str(path))
            if pix.isNull():
                # Si no se puede cargar, no bloquea la app
                return
            # Escalado suave manteniendo proporción
            pix = pix.scaledToHeight(target_h, Qt.SmoothTransformation)
            self.logo_lbl.setPixmap(pix)
            self.logo_lbl.setFixedSize(pix.size())
        except Exception:
            pass

    # ------------------------------------------------------------
    # Acciones principales de la interfaz
    # ------------------------------------------------------------
    def on_synthesize(self):
        """
            Acción: iniciar síntesis de voz.
              - Lee texto y valores de sliders
              - Valida entrada
              - Lanza el worker de síntesis en segundo plano
              - Desactiva botón y añade log de "Preparando..."
        """
        # Texto y modo
        text = self.text_edit.text().strip()
        mode = self.mode_combo.currentData()
        if not text:
            QMessageBox.warning(self, "Ups", "Escribe un texto para sintetizar.")
            return

        # Leer valores de sliders desde las cajitas (como en tu diseño original)
        try:
            g = float(self.gender_box.text())
            a = float(self.age_box.text())
            p = float(self.pitch_box.text())
        except Exception:
            QMessageBox.warning(self, "Ups", "Valores de sliders no válidos.")
            return

        # Idioma seleccionado
        lang = self.lang_combo.currentData()

        # Feedback en UI
        self.btn_synth.setEnabled(False)
        self._append_log("Preparando...")

        # Crear worker de síntesis
        self.worker = SynthesisWorker(
            language=lang,
            text=text,
            gender=g,
            age=a,
            out_dir=Path("./out_wavs"),
            pitch=p,
            mode=mode
        )
        # Conectar señales del worker a UI
        self.worker.status.connect(self._append_log)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_done(self, path_str: str):
        """
            Acción cuando la síntesis termina correctamente:
              - Guarda la ruta del WAV generado
              - Añade log de "OK"
              - Reactiva botón de sintetizar y activa reproducir
        """
        self.last_wav = path_str
        self._append_log(f"OK: {path_str}")
        self.btn_synth.setEnabled(True)
        self.btn_play.setEnabled(True)

    def on_failed(self, msg: str):
        """
            Acción cuando la síntesis falla:
              - Reactiva botón de sintetizar
              - Desactiva botón de reproducir
              - Añade log de error
              - Muestra cuadro emergente con detalle
        """
        self.btn_synth.setEnabled(True)
        self.btn_play.setEnabled(False)
        self._append_log("Fallo en síntesis.")
        QMessageBox.critical(self, "Error", msg)

    def on_play(self):
        """
            Acción: reproducir último audio generado.
              - Configura volumen
              - Carga archivo WAV
              - Lanza reproducción
        """
        if not self.last_wav:
            return
        self.audio_out.setVolume(0.85)
        self.player.setSource(QUrl.fromLocalFile(self.last_wav))
        self.player.play()

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    apply_pretty_theme(app)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
