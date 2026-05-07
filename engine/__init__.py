"""
engine — Motor de renderização 3D software.

Pacote que reúne os 4 módulos fundamentais do pipeline gráfico:
  - transforms:  Biblioteca de transformações geométricas (matrizes 4×4)
  - quaternion:  Quatérnios de Hamilton para rotações suaves
  - camera:      Câmera virtual com projeção perspectiva
  - renderer:    Pipeline de renderização (Bresenham + Painter)
  - mesh:        Objetos 3D (Objeto3D + Mesh)
"""

# ── Parte 1: Transformações ──────────────────────────────────────────────────
from engine.transforms import (
    translacao,
    escala,
    rotacao_x,
    rotacao_y,
    rotacao_z,
    cisalhamento_xy,
    compor,
)

# ── Parte 2: Quatérnios ─────────────────────────────────────────────────────
from engine.quaternion import Quaternion, slerp

# ── Parte 3: Câmera ─────────────────────────────────────────────────────────
from engine.camera import Camera

# ── Parte 4: Objetos 3D e Pipeline ───────────────────────────────────────────
from engine.renderer import Renderizador
from engine.mesh import Objeto3D, Mesh, FORMAS

__all__ = [
    # Transformações
    "translacao", "escala", "rotacao_x", "rotacao_y", "rotacao_z",
    "cisalhamento_xy", "compor",
    # Quatérnios
    "Quaternion", "slerp",
    # Câmera
    "Camera",
    # Pipeline
    "Renderizador", "Objeto3D", "Mesh", "FORMAS",
]
