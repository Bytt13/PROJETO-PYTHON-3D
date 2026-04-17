"""
=============================================================================
 PARTE 3: CÂMERA VIRTUAL E PROJEÇÃO PERSPECTIVA
=============================================================================

 ╔══════════════════════════════════════════════════════════════════╗
 ║  A ANALOGIA DO PROJETOR DE SLIDES                               ║
 ╚══════════════════════════════════════════════════════════════════╝

 Renderizar uma cena 3D em uma tela 2D é EXATAMENTE como um projetor:

   [Cena 3D] → [Slide/Câmera] → [Projeção] → [Parede/Tela]
   
   1. A CÂMERA é o projetor: ela tem posição, direção que olha e
      uma "cabeça" (vetor UP) que define o que é "para cima".
      
   2. A MATRIZ DE VISÃO (View Matrix) é como "carregar" o slide no projetor:
      ela move o MUNDO INTEIRO para que a câmera fique na origem,
      olhando para o eixo -Z. Mais fácil calcular assim!
      
   3. A MATRIZ DE PROJEÇÃO é a lente do projetor:
      ela determina o FOV (campo de visão), quanto das bordas você enxerga,
      e — mais importante — comprime objetos distantes para parecerem menores
      (perspectiva!).

 ╔══════════════════════════════════════════════════════════════════╗
 ║  COMO A PERSPECTIVA CRIA ILUSÃO DE PROFUNDIDADE?                ║
 ╚══════════════════════════════════════════════════════════════════╝

 No mundo real, objetos distantes parecem menores porque a luz de um objeto
 longe chega em ângulo mais fechado ao nosso olho.

 Na matemática, simulamos isso com a DIVISÃO POR W:
   Depois da matriz de projeção, x_tela = x_homogêneo / w
   
 O 'w' contém a profundidade do objeto. Objetos distantes têm w grande,
 então a divisão os ENCOLHE. Objetos próximos têm w ~= 1, ficam grandes.
 
 É a mesma razão pela qual trilhos de trem "convergem" no horizonte:
 matematicamente, a distância vai para o infinito, mas na tela converge.

 ╔══════════════════════════════════════════════════════════════════╗
 ║  O SISTEMA UVN (BASE DA CÂMERA)                                 ║
 ╚══════════════════════════════════════════════════════════════════╝

 A câmera define seu próprio sistema de coordenadas com 3 vetores:
   n = direção para ONDE a câmera olha (eixo Z da câmera)
   u = direção da DIREITA da câmera (eixo X da câmera)
   v = direção de CIMA da câmera (eixo Y da câmera)

 ANALOGIA: Você olhando em frente:
   n = o seu nariz (frente/trás)
   u = seus ombros (direita/esquerda)
   v = seu topo da cabeça (cima/baixo)

=============================================================================
"""

import numpy as np


class Camera:
    """
    Câmera virtual 3D com sistema de coordenadas UVN.

    Gerencia posição, orientação e parâmetros de projeção.
    """

    def __init__(
        self,
        posicao: np.ndarray = None,
        alvo: np.ndarray = None,
        up: np.ndarray = None,
        fov: float = 60.0,
        aspect_ratio: float = 16/9,
        z_near: float = 0.1,
        z_far: float = 100.0
    ):
        """
        Inicializa a câmera.

        Args:
            posicao: onde a câmera está no mundo [x, y, z]
            alvo: para onde a câmera está olhando [x, y, z]
            up: vetor "para cima" do mundo (geralmente [0,1,0])
            fov: field of view (campo de visão) em graus
            aspect_ratio: proporção largura/altura da tela
            z_near: distância do plano de recorte próximo (mínimo visível)
            z_far: distância do plano de recorte distante (máximo visível)
        """
        self.posicao = np.array(posicao if posicao is not None else [0, 2, 8], dtype=float)
        self.alvo = np.array(alvo if alvo is not None else [0, 0, 0], dtype=float)
        self.up_mundo = np.array(up if up is not None else [0, 1, 0], dtype=float)

        self.fov = float(fov)             # Field of View em graus
        self.aspect_ratio = float(aspect_ratio)
        self.z_near = float(z_near)
        self.z_far = float(z_far)

        # Velocidades de movimentação
        self.velocidade = 0.15
        self.sensibilidade_mouse = 0.003

        # Ângulos de orientação (yaw = esquerda/direita, pitch = cima/baixo)
        self.yaw = -np.pi / 2    # começa olhando para -Z
        self.pitch = -0.2

    # ── Sistema de coordenadas UVN ────────────────────────────────────────────

    def _calcular_uvn(self):
        """
        Calcula os vetores u, v, n da câmera a partir de yaw e pitch.

        PROCESSO (regra da mão direita):
          1. n = direção do olhar (frente)
          2. u = n × up_mundo (direita — produto vetorial)
          3. v = u × n (cima da câmera, recalculado para ser ortogonal a n)
        
        Por que usar produto vetorial?
        O produto vetorial de dois vetores dá um terceiro vetor perpendicular
        a ambos — perfeito para construir uma base ortogonal!
        """
        # Vetor 'n' (frente): calculado a partir dos ângulos de Euler da câmera
        n = np.array([
            np.cos(self.pitch) * np.cos(self.yaw),
            np.sin(self.pitch),
            np.cos(self.pitch) * np.sin(self.yaw)
        ], dtype=float)
        n = n / np.linalg.norm(n)  # normaliza

        # Vetor 'u' (direita): perpendicular a n e ao up do mundo
        u = np.cross(n, self.up_mundo)
        norm_u = np.linalg.norm(u)
        if norm_u < 1e-10:
            # Câmera apontando direto para cima/baixo — avoid gimbal lock
            u = np.array([1, 0, 0], dtype=float)
        else:
            u = u / norm_u

        # Vetor 'v' (cima da câmera): perpendicular a u e n
        v = np.cross(u, n)
        v = v / np.linalg.norm(v)

        return u, v, n

    @property
    def frente(self) -> np.ndarray:
        """Vetor unitário na direção que a câmera olha."""
        _, _, n = self._calcular_uvn()
        return n

    @property
    def direita(self) -> np.ndarray:
        """Vetor unitário para a direita da câmera."""
        u, _, _ = self._calcular_uvn()
        return u

    # ── Matrizes de transformação ─────────────────────────────────────────────

    def get_view_matrix(self) -> np.ndarray:
        """
        Calcula a Matriz de Visão (View Matrix / LookAt Matrix).

        ANALOGIA: Imagine que você está filmando uma cena.
        A View Matrix "move o mundo inteiro" para que a câmera
        fique na ORIGEM do espaço, olhando para o eixo -Z.
        
        Isso simplifica enormemente todos os cálculos subsequentes:
        ao invés de sempre considerar onde a câmera está, tratamos
        ela como fixa e movemos tudo ao redor dela.

        A matriz é construída em 2 partes:
          1. Rotação: alinha os eixos do mundo com os eixos da câmera (u,v,n)
          2. Translação: move a cena para que a câmera fique na origem

        Combinadas em uma só operação: View = Rotação × Translação_inversa

        Returns:
            Matriz 4×4 de visão
        """
        u, v, n = self._calcular_uvn()
        p = self.posicao

        # Parte de rotação: cada eixo da câmera vira uma linha da matriz
        # O produto escalar com -p faz a translação inversa
        return np.array([
            [u[0], u[1], u[2], -np.dot(u, p)],
            [v[0], v[1], v[2], -np.dot(v, p)],
            [n[0], n[1], n[2], -np.dot(n, p)],
            [   0,    0,    0,              1]
        ], dtype=float)

    def get_projection_matrix(self) -> np.ndarray:
        """
        Calcula a Matriz de Projeção Perspectiva.

        ANALOGIA DA PROFUNDIDADE:
        Imagine olhar por um vitral. Linhas que vão para longe
        parecem convergir em um "ponto de fuga". Isso é perspectiva.

        O FRUSTO (Frustum) é o "cone" do que a câmera enxerga:
          - Plano próximo (near): tudo mais próximo que isso é invisível
          - Plano longe (far): tudo mais distante é invisível
          - FOV: o ângulo de abertura do cone (campo de visão)
          
        Quanto maior o FOV (ex: 90°), mais "wide angle" a câmera parece.
        Quanto menor (ex: 30°), efeito "zoom" ou "telescópio".

        COMO O DEPTH FUNCIONA:
        A matriz mapeia z_near para -1 e z_far para +1 (NDC).
        No passo de "divisão por W" depois, isso cria a perspectiva real.

           f = 1 / tan(FOV/2)
           
        [f/a    0       0           0     ]
        [ 0     f       0           0     ]
        [ 0     0  (far+near)/(near-far)  2*far*near/(near-far)]
        [ 0     0      -1           0     ]

        Returns:
            Matriz 4×4 de projeção perspectiva
        """
        fov_rad = np.radians(self.fov)
        f = 1.0 / np.tan(fov_rad / 2.0)
        n = self.z_near
        fa = self.z_far
        a = self.aspect_ratio

        return np.array([
            [f / a, 0,               0,                      0],
            [    0, f,               0,                      0],
            [    0, 0, (fa + n) / (n - fa), (2 * fa * n) / (n - fa)],
            [    0, 0,              -1,                      0]
        ], dtype=float)

    # ── Movimentação da câmera ────────────────────────────────────────────────

    def mover_frente(self, quantidade: float = None):
        """Move a câmera na direção que ela olha."""
        v = quantidade if quantidade is not None else self.velocidade
        self.posicao += self.frente * v

    def mover_tras(self, quantidade: float = None):
        """Recua a câmera."""
        v = quantidade if quantidade is not None else self.velocidade
        self.posicao -= self.frente * v

    def mover_direita(self, quantidade: float = None):
        """Move a câmera para a sua direita."""
        v = quantidade if quantidade is not None else self.velocidade
        self.posicao += self.direita * v

    def mover_esquerda(self, quantidade: float = None):
        """Move a câmera para a sua esquerda."""
        v = quantidade if quantidade is not None else self.velocidade
        self.posicao -= self.direita * v

    def mover_cima(self, quantidade: float = None):
        """Move a câmera para cima (eixo Y do mundo)."""
        v = quantidade if quantidade is not None else self.velocidade
        self.posicao[1] += v

    def mover_baixo(self, quantidade: float = None):
        """Move a câmera para baixo (eixo Y do mundo)."""
        v = quantidade if quantidade is not None else self.velocidade
        self.posicao[1] -= v

    def rotacionar(self, delta_yaw: float, delta_pitch: float):
        """
        Rotaciona a câmera (olhar) com controle de yaw e pitch.

        yaw: rotação horizontal (esquerda/direita)
        pitch: rotação vertical (cima/baixo)

        Limita pitch para evitar gimbal lock quando olha 90° acima/abaixo.
        """
        self.yaw += delta_yaw
        self.pitch += delta_pitch
        # Limita pitch: evita virar de cabeça para baixo (-85° a +85°)
        self.pitch = np.clip(self.pitch, -np.radians(85), np.radians(85))

    def zoom(self, delta_fov: float):
        """
        Altera o FOV para simular zoom.

        Diminuir FOV → efeito telescópio (zoom in)
        Aumentar FOV → efeito wide-angle (zoom out)
        """
        self.fov = np.clip(self.fov + delta_fov, 10.0, 120.0)
