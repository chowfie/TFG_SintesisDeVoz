import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main():
    # Argumentos
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Ruta al merged_per_speaker.csv")
    ap.add_argument("--out_dir", required=True, help="Carpeta de salida para scaler, pca y coords")
    ap.add_argument("--n_components", type=int, default=64)
    args = ap.parse_args()

    # Rutas y carpeta de salida
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cargar embeddings por speaker
    df = pd.read_csv(csv_path)
    # Detectar columnas de embedding e0..eN
    emb_cols = [c for c in df.columns if c.startswith("e")]
    # Matriz X: (n_speakers x d_emb)
    X = df[emb_cols].to_numpy(dtype=np.float32)

    # Estandarizar (media 0, var 1)
    # Importante para que PCA no se sesgue por escalas distintas
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Ajustar PCA y proyectar
    pca = PCA(n_components=args.n_components, random_state=0)
    # Z: coordenadas comprimidas (n_speakers x k)
    Z = pca.fit_transform(Xs)

    # Métrica rápida de fidelidad (reconstrucción)
    # Reconstruimos X desde Z y medimos similitud coseno original vs reconstruido
    Xs_rec = pca.inverse_transform(Z)
    X_rec = scaler.inverse_transform(Xs_rec)
    num = np.sum(X * X_rec, axis=1)
    den = np.linalg.norm(X, axis=1) * np.linalg.norm(X_rec, axis=1)
    cos = num / np.clip(den, 1e-9, None)

    # Guardar artefactos y coordenadas
    joblib.dump(scaler, out_dir / "scaler.joblib")   # para transformar futuros embeddings
    joblib.dump(pca,    out_dir / "pca.joblib")      # PCA entrenado
    # CSV con z1..zk (+ columnas de identificación si existen)
    coords = pd.DataFrame(Z, columns=[f"z{i+1}" for i in range(args.n_components)])
    if "speaker_id" in df.columns: coords.insert(0, "speaker_id", df["speaker_id"].values)
    if "corpus" in df.columns:      coords.insert(0, "corpus", df["corpus"].values)
    coords.to_csv(out_dir / "pca_coordinates.csv", index=False)

    # Resumen
    print(f"n={X.shape[0]}, d={X.shape[1]}, k={args.n_components}")
    print(f"Varianza explicada total: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"Coseno medio: {cos.mean():.4f}  p5: {np.percentile(cos,5):.4f}  min: {cos.min():.4f}")
    print("Guardado en:", out_dir)

if __name__ == "__main__":
    main()
