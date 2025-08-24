import os
import shutil
import pandas as pd

# ------------------------------------------------------------
# Rutas
# ------------------------------------------------------------
base_path = r"C:\Users\sreal\Desktop\TFG\CODIGO\LJSpeech-1.1"
wavs_path = os.path.join(base_path, "wavs")
metadata_path = os.path.join(base_path, "metadata.csv")

# Ruta donde guardar el conjunto dividido
split_path = r"/CODIGO/LJSPEECH"
train_path = os.path.join(split_path, "train")
test_path = os.path.join(split_path, "test")

# Crear carpetas si no existen
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Leer metadatos
df = pd.read_csv(metadata_path, sep="|", header=None, names=["file", "text", "normalized_text"])

# Filtrar por archivos
wav_files = sorted(os.listdir(wavs_path))
df = df[df["file"].apply(lambda x: x + ".wav" in wav_files)]

# Dividir: 250 entrenamiento + 28 test
train_df = df[:250]
test_df = df[250:278]

# Copiar audios de entrenamiento
for _, row in train_df.iterrows():
    src = os.path.join(wavs_path, row["file"] + ".wav")
    dst = os.path.join(train_path, row["file"] + ".wav")
    shutil.copy(src, dst)
train_df.to_csv(os.path.join(train_path, "metadata.csv"), sep="|", index=False, header=False)

# Copiar audios de test
for _, row in test_df.iterrows():
    src = os.path.join(wavs_path, row["file"] + ".wav")
    dst = os.path.join(test_path, row["file"] + ".wav")
    shutil.copy(src, dst)
test_df.to_csv(os.path.join(test_path, "metadata.csv"), sep="|", index=False, header=False)

print("✅ División completada:")
print(f"- Entrenamiento: {len(train_df)} audios")
print(f"- Test: {len(test_df)} audios")
print(f"- Carpeta creada: {split_path}")
