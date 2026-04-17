"""
mesh.py — Definição das malhas 3D (vértices + faces poligonais).

Cada sólido é criado com:
  - vertices:  array Nx3 em coordenadas locais (centrado na origem)
  - faces:     lista de tuplas de índices de vértices (polígono)
  - cor:       (R, G, B) base para o shading
  - nome / descricao / propriedades: metadados para o menu
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any


@dataclass
class Mesh:
    nome: str
    descricao: str          # texto multi-linha (use \n)
    vertices: np.ndarray    # (N, 3) float64
    faces: List[Tuple]      # [(i,j,k,...), ...]
    cor: Tuple[int,int,int]
    icone: str
    propriedades: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Utilitários
# ─────────────────────────────────────────────────────────────────────────────

def _norm_verts(V: np.ndarray) -> np.ndarray:
    """Projeta todos os vértices para a esfera unitária."""
    n = np.linalg.norm(V, axis=1, keepdims=True)
    return V / np.where(n < 1e-10, 1, n)


# ─────────────────────────────────────────────────────────────────────────────
# Sólidos
# ─────────────────────────────────────────────────────────────────────────────

def criar_cubo(s: float = 1.0) -> Mesh:
    """Hexaedro regular — 6 faces quadradas."""
    V = np.array([
        [-s,-s,-s],[s,-s,-s],[s,s,-s],[-s,s,-s],   # frente: 0-3
        [-s,-s, s],[s,-s, s],[s,s, s],[-s,s, s],   # trás:   4-7
    ], dtype=float)

    # Faces com winding counterclockwise visto de fora
    F = [
        (3,2,1,0),  # frente  (-z)
        (4,5,6,7),  # trás    (+z)
        (0,4,7,3),  # esquerda(-x)
        (1,2,6,5),  # direita (+x)
        (3,7,6,2),  # topo    (+y)
        (0,1,5,4),  # base    (-y)
    ]
    return Mesh(
        nome="Cubo",
        descricao="O Hexaedro Regular.\nTodas as 6 faces são quadrados\nperfeitos e congruentes.\nO mais familiar dos Sólidos de Platão.",
        vertices=V, faces=F, cor=(0,210,240), icone="⬛",
        propriedades={"Vértices":8,"Faces":6,"Arestas":12,"Tipo":"Sólido de Platão"},
    )


def criar_tetraedro() -> Mesh:
    """4 faces triangulares — o mais simples dos sólidos de Platão."""
    a = 2 * np.sqrt(2) / 3
    V = np.array([
        [0, 1, 0],
        [a, -1/3, 0],
        [-a/2, -1/3,  np.sqrt(6)/3],
        [-a/2, -1/3, -np.sqrt(6)/3],
    ], dtype=float)
    F = [
        (0,1,2),
        (0,2,3),
        (0,3,1),
        (3,2,1),   # base
    ]
    return Mesh(
        nome="Tetraedro",
        descricao="O sólido de Platão mais simples.\nApenas 4 faces triangulares.\nSímbolo do elemento Fogo\nna filosofia grega clássica.",
        vertices=V, faces=F, cor=(255,90,50), icone="△",
        propriedades={"Vértices":4,"Faces":4,"Arestas":6,"Tipo":"Sólido de Platão"},
    )


def criar_octaedro() -> Mesh:
    """8 faces triangulares — dual do cubo."""
    V = np.array([
        [0, 1, 0], [0,-1, 0],   # polo norte/sul
        [1, 0, 0], [-1,0, 0],   # eixo x
        [0, 0, 1], [0, 0,-1],   # eixo z
    ], dtype=float)
    F = [
        (0,2,4),(0,4,3),(0,3,5),(0,5,2),  # hemisfério norte
        (1,4,2),(1,3,4),(1,5,3),(1,2,5),  # hemisfério sul
    ]
    return Mesh(
        nome="Octaedro",
        descricao="8 faces triangulares equiláteras.\nSímbolo do elemento Ar.\nDual do Cubo — centros das\nfaces formam um cubo.",
        vertices=V, faces=F, cor=(40,220,140), icone="◇",
        propriedades={"Vértices":6,"Faces":8,"Arestas":12,"Tipo":"Sólido de Platão"},
    )


def criar_icosaedro() -> Mesh:
    """20 faces triangulares — o mais esférico entre os sólidos de Platão."""
    phi = (1 + np.sqrt(5)) / 2   # razão áurea ≈ 1.618
    V = _norm_verts(np.array([
        [-1, phi,0],[1,phi,0],[-1,-phi,0],[1,-phi,0],
        [0,-1,phi],[0,1,phi],[0,-1,-phi],[0,1,-phi],
        [phi,0,-1],[phi,0,1],[-phi,0,-1],[-phi,0,1],
    ], dtype=float))
    F = [
        (0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
        (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
        (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
        (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1),
    ]
    return Mesh(
        nome="Icosaedro",
        descricao="20 faces triangulares equiláteras.\nA mais esférica entre os Sólidos\nde Platão. Base do design de\ndômas geodésicas (Buckminster Fuller).",
        vertices=V, faces=F, cor=(170,60,255), icone="⬡",
        propriedades={"Vértices":12,"Faces":20,"Arestas":30,"Tipo":"Sólido de Platão"},
    )


def criar_piramide() -> Mesh:
    """Pirâmide de base quadrada."""
    b, h = 1.0, 1.6
    V = np.array([
        [-b,0,-b],[b,0,-b],[b,0,b],[-b,0,b],  # base
        [0, h, 0],                              # apex
    ], dtype=float)
    F = [
        (3,2,1,0),  # base (voltada para baixo)
        (0,1,4),    # frente
        (1,2,4),    # direita
        (2,3,4),    # trás
        (3,0,4),    # esquerda
    ]
    return Mesh(
        nome="Pirâmide",
        descricao="Base quadrada e 4 faces\ntriangulares convergindo no ápice.\nForma das Pirâmides do Egito\ne da Mesoamérica.",
        vertices=V, faces=F, cor=(255,205,30), icone="⛛",
        propriedades={"Vértices":5,"Faces":5,"Arestas":8,"Tipo":"Pirâmide Quadrada"},
    )


def criar_prisma_triangular() -> Mesh:
    """Prisma com base triangular equiláteral."""
    r, h = 1.0, 1.1
    top = [[r*np.cos(np.radians(a+90)), h, r*np.sin(np.radians(a+90))] for a in [0,120,240]]
    bot = [[r*np.cos(np.radians(a+90)),-h, r*np.sin(np.radians(a+90))] for a in [0,120,240]]
    V = np.array(top + bot, dtype=float)  # 0,1,2=top | 3,4,5=bot
    F = [
        (2,1,0),        # topo
        (3,4,5),        # base
        (0,1,4,3),      # lateral 1
        (1,2,5,4),      # lateral 2
        (2,0,3,5),      # lateral 3
    ]
    return Mesh(
        nome="Prisma Triangular",
        descricao="Duas bases triangulares paralelas\nunidas por 3 faces retangulares.\nUsado em prismas ópticos para\ndecompor a luz em espectros.",
        vertices=V, faces=F, cor=(60,150,255), icone="▷",
        propriedades={"Vértices":6,"Faces":5,"Arestas":9,"Tipo":"Prisma"},
    )


def criar_esfera(stacks: int = 16, slices: int = 24) -> Mesh:
    """Esfera UV — gerada por revolução de semicírculo."""
    verts, faces = [], []
    for i in range(stacks + 1):
        lat = np.pi * (-0.5 + i / stacks)
        for j in range(slices):
            lon = 2 * np.pi * j / slices
            verts.append([
                np.cos(lat) * np.cos(lon),
                np.sin(lat),
                np.cos(lat) * np.sin(lon),
            ])
    for i in range(stacks):
        for j in range(slices):
            a = i * slices + j
            b = i * slices + (j+1) % slices
            c = (i+1) * slices + (j+1) % slices
            d = (i+1) * slices + j
            faces.append((a, b, c, d))

    n_v = (stacks+1)*slices
    return Mesh(
        nome="Esfera",
        descricao="Gerada por revolução de um\nsemicírculo ao redor de um eixo.\nTodos os pontos equidistantes\ndo centro (raio constante = 1).",
        vertices=np.array(verts, dtype=float),
        faces=faces, cor=(255,80,170), icone="●",
        propriedades={"Vértices":n_v,"Faces":stacks*slices,"Arestas":"~","Tipo":"Quádrica"},
    )


def criar_toro(R: float = 1.0, r: float = 0.40, stacks: int = 28, slices: int = 18) -> Mesh:
    """
    Toro (rosca) — superfície de revolução de um círculo.
    R = raio maior (centro do buraco ao centro do tubo)
    r = raio menor (raio do tubo)
    """
    verts, faces = [], []
    for i in range(stacks):
        u = 2 * np.pi * i / stacks
        for j in range(slices):
            v = 2 * np.pi * j / slices
            x = (R + r * np.cos(v)) * np.cos(u)
            y = r * np.sin(v)
            z = (R + r * np.cos(v)) * np.sin(u)
            verts.append([x, y, z])
    for i in range(stacks):
        for j in range(slices):
            a = i * slices + j
            b = i * slices + (j+1) % slices
            c = ((i+1) % stacks) * slices + (j+1) % slices
            d = ((i+1) % stacks) * slices + j
            faces.append((a, d, c, b))   # winding para normal outward

    n_v = stacks * slices
    return Mesh(
        nome="Toro",
        descricao="Superfície gerada pela revolução\nde um círculo ao redor de um\neixo externo. Topologia diferente\ndas esferas: tem 'buraco' no centro.",
        vertices=np.array(verts, dtype=float),
        faces=faces, cor=(255,135,0), icone="◎",
        propriedades={"Vértices":n_v,"Faces":n_v,"Arestas":"~","Tipo":"Superfície de Revolução"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Catálogo de formas disponíveis no menu
# ─────────────────────────────────────────────────────────────────────────────
FORMAS: List[Mesh] = [
    criar_cubo(),
    criar_tetraedro(),
    criar_octaedro(),
    criar_icosaedro(),
    criar_piramide(),
    criar_prisma_triangular(),
    criar_esfera(),
    criar_toro(),
]
