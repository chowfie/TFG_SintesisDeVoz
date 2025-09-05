# -*- coding: utf-8 -*-
"""
Figura conceptual de speaker embeddings en 3D (sin emojis).
Guarda el PNG en el directorio actual como 'speaker_embeddings_demo.png'.
Requisitos: numpy, matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Semilla para reproducibilidad
rng = np.random.default_rng(7)

# Definimos 4 clústeres conceptuales (p. ej., género × tono)
clusters = [
    {"name": "Femenina · voz aguda",  "abbr": "F-aguda",  "center": np.array([ 2.2,  2.0,  1.6]), "n": 10, "color": "#4C78A8", "marker": "o"},
    {"name": "Femenina · voz grave",  "abbr": "F-grave",  "center": np.array([ 1.8, -2.2,  0.6]), "n": 10, "color": "#F58518", "marker": "s"},
    {"name": "Masculina · voz aguda", "abbr": "M-aguda",  "center": np.array([-2.0,  1.6, -0.8]), "n": 10, "color": "#54A24B", "marker": "^"},
    {"name": "Masculina · voz grave", "abbr": "M-grave",  "center": np.array([-2.4, -2.0, -1.8]), "n": 10, "color": "#E45756", "marker": "D"},
]

# Generamos puntos alrededor de cada centro
all_points   = []
all_colors   = []
all_markers  = []
all_labels   = []
for c in clusters:
    pts = c["center"] + rng.normal(scale=0.35, size=(c["n"], 3))
    all_points.append(pts)
    all_colors += [c["color"]]  * c["n"]
    all_markers += [c["marker"]] * c["n"]
    all_labels  += [c["abbr"]]   * c["n"]

points = np.vstack(all_points)

# Figura y ejes 3D
fig = plt.figure(figsize=(9, 7), dpi=160)
ax = fig.add_subplot(111, projection="3d")
ax.set_title("Speaker embeddings (conceptual) con rasgos de voz", pad=16)

# Dibujamos cada clúster con su color/marker
idx = 0
for c in clusters:
    n = c["n"]
    subset = points[idx:idx+n]
    ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2],
               s=38, c=c["color"], marker=c["marker"], alpha=0.95,
               label=c["name"], depthshade=False, edgecolors="none")
    # Etiqueta corta encima de algunos puntos (1 de cada 3) para legibilidad
    for j in range(0, n, 3):
        x, y, z = subset[j]
        ax.text(x, y, z + 0.18, c["abbr"], color=c["color"],
                fontsize=8, ha="center", va="bottom")
    idx += n

# Etiquetas de ejes
ax.set_xlabel("Dim 1", labelpad=8)
ax.set_ylabel("Dim 2", labelpad=8)
ax.set_zlabel("Dim 3", labelpad=8)

# Fondo suave (sin depender de fuentes especiales)
ax.xaxis.pane.set_facecolor((0.96, 0.96, 0.96, 0.85))
ax.yaxis.pane.set_facecolor((0.96, 0.96, 0.96, 0.85))
ax.zaxis.pane.set_facecolor((0.96, 0.96, 0.96, 0.85))
ax.grid(False)

# Flechas conceptuales (ejes semánticos orientativos)
def arrow3d(start, vec, color, text):
    ax.quiver(*start, *vec, arrow_length_ratio=0.09, color=color, linewidth=2)
    tip = start + vec
    ax.text(tip[0], tip[1], tip[2] + 0.05, text, color=color, fontsize=10, ha="left", va="bottom")

arrow3d(np.array([0, -3.0, -2.2]), np.array([0, 6.2, 0]), "#555555", "tono (grave → agudo)")
arrow3d(np.array([-3.2, 0, -2.2]), np.array([6.4, 0, 0]), "#555555", "edad/masculinidad ↔ femineidad")

# Leyenda clara
leg = ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.96), frameon=True)
leg.get_frame().set_alpha(0.9)

plt.tight_layout()
out_path = "speaker_embeddings_demo.png"  # se guarda en el directorio actual
plt.savefig(out_path, bbox_inches="tight")
print(f"Imagen guardada en: {out_path}")
