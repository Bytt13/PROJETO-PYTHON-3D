"""
=============================================================================
 PARTE 5: APLICAÇÃO PRINCIPAL — VISUALIZADOR 3D INTERATIVO
=============================================================================

 Como executar:
   python main.py

 Controles:
   W/S         → mover câmera frente/trás
   A/D         → mover câmera esquerda/direita
   Q/E         → mover câmera baixo/cima
   ↑↓←→        → girar câmera (pitch/yaw)
   Z/X         → zoom (diminuir/aumentar FOV)
   R           → resetar câmera
   ESPAÇO      → pausar/retomar animação
   ESC         → sair

=============================================================================
"""

import sys
import math
import numpy as np
import pygame

from transforms import (
    translacao, escala, rotacao_x, rotacao_y, rotacao_z,
    cisalhamento_xy, compor
)
from quaternion import Quaternion, slerp
from camera import Camera
from renderer import Renderizador


# ════════════════════════════════════════════════════════════════════════════
# GEOMETRIAS 3D
# ════════════════════════════════════════════════════════════════════════════

def criar_cubo(tamanho: float = 1.0):
    """
    Cria os vértices e arestas de um cubo centrado na origem.

    ANATOMIA DO CUBO:
    8 vértices nos cantos, 12 arestas (4 por face lateral, 4 topo, 4 baixo)

         7────6
        /|   /|
       4────5 |
       | 3──|─2
       |/   |/
       0────1

    Args:
        tamanho: metade do lado do cubo (raio)

    Returns:
        (vertices, arestas) com coordenadas no espaço local
    """
    s = tamanho
    vertices = np.array([
        [-s, -s, -s],  # 0: frente-baixo-esquerda
        [ s, -s, -s],  # 1: frente-baixo-direita
        [ s, -s,  s],  # 2: trás-baixo-direita
        [-s, -s,  s],  # 3: trás-baixo-esquerda
        [-s,  s, -s],  # 4: frente-cima-esquerda
        [ s,  s, -s],  # 5: frente-cima-direita
        [ s,  s,  s],  # 6: trás-cima-direita
        [-s,  s,  s],  # 7: trás-cima-esquerda
    ], dtype=float)

    arestas = [
        # Base inferior
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Base superior
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Pilares verticais
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    return vertices, arestas


def criar_piramide(base: float = 1.0, altura: float = 1.5):
    """
    Cria uma pirâmide de base quadrada.

    ANATOMIA:
    4 vértices na base + 1 vértice no topo = 5 vértices
    8 arestas de aresta (4 base + 4 lados)

         apex (4)
        /|\\
       / | \\
      /  |  \\
     0───1───2
     └───3───┘
     
    Args:
        base: metade do lado da base
        altura: altura do apex acima da origem

    Returns:
        (vertices, arestas)
    """
    b = base
    h = altura
    vertices = np.array([
        [-b, 0, -b],   # 0: base frente-esquerda
        [ b, 0, -b],   # 1: base frente-direita
        [ b, 0,  b],   # 2: base trás-direita
        [-b, 0,  b],   # 3: base trás-esquerda
        [ 0, h,  0],   # 4: apex (topo)
    ], dtype=float)

    arestas = [
        # Base
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Arestas laterais do apex
        (0, 4), (1, 4), (2, 4), (3, 4),
    ]

    return vertices, arestas


def criar_grid(tamanho: int = 6, passo: float = 1.0):
    """
    Cria um grid plano no plano XZ para referência visual.

    ANALOGIA: é o "chão" do nosso mundo 3D.

    Args:
        tamanho: quantidade de linhas de cada lado do centro
        passo: distância entre linhas

    Returns:
        (vertices, arestas)
    """
    vertices = []
    arestas = []
    v_idx = 0

    for i in range(-tamanho, tamanho + 1):
        # Linha paralela ao Z
        vertices.append([i * passo, 0, -tamanho * passo])
        vertices.append([i * passo, 0,  tamanho * passo])
        arestas.append((v_idx, v_idx + 1))
        v_idx += 2

        # Linha paralela ao X
        vertices.append([-tamanho * passo, 0, i * passo])
        vertices.append([ tamanho * passo, 0, i * passo])
        arestas.append((v_idx, v_idx + 1))
        v_idx += 2

    return np.array(vertices, dtype=float), arestas


def criar_eixos(comprimento: float = 2.0):
    """
    Cria as setas dos eixos X (vermelho), Y (verde), Z (azul).
    Retorna lista de (vertices, arestas, cor) para cada eixo.
    """
    eixos = [
        (np.array([[0,0,0],[comprimento,0,0]], dtype=float), [(0,1)], (220, 60, 60)),    # X: vermelho
        (np.array([[0,0,0],[0,comprimento,0]], dtype=float), [(0,1)], (60, 220, 60)),    # Y: verde
        (np.array([[0,0,0],[0,0,comprimento]], dtype=float), [(0,1)], (60, 60, 220)),    # Z: azul
    ]
    return eixos


# ════════════════════════════════════════════════════════════════════════════
# PALETA DE CORES
# ════════════════════════════════════════════════════════════════════════════

PRETO           = (  8,   8,  18)
BRANCO          = (255, 255, 255)
CIANO           = ( 0,  220, 255)
LARANJA         = (255, 140,   0)
VIOLETA         = (180,  80, 255)
VERDE_NEON      = (  0, 255, 140)
AMARELO         = (255, 230,  60)
CINZA_GRADE     = ( 45,  45,  65)
CINZA_TEXTO     = (180, 180, 210)
AZUL_ESCURO     = ( 10,  10,  35)
VERMELHO_GLOW   = (255,  60,  60)


# ════════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS DE HUD (Interface Heads-Up Display)
# ════════════════════════════════════════════════════════════════════════════

class HUD:
    """Gerencia o display de texto na tela (overlay 2D)."""

    def __init__(self, superficie: pygame.Surface):
        self.sup = superficie
        self.w = superficie.get_width()
        self.h = superficie.get_height()

        # Fontes
        pygame.font.init()
        self.fonte_titulo = pygame.font.SysFont("Consolas", 15, bold=True)
        self.fonte_info   = pygame.font.SysFont("Consolas", 13)
        self.fonte_small  = pygame.font.SysFont("Consolas", 11)

    def _texto(self, texto: str, x: int, y: int, cor=CINZA_TEXTO, fonte=None):
        if fonte is None:
            fonte = self.fonte_info
        surf = fonte.render(texto, True, cor)
        self.sup.blit(surf, (x, y))

    def desenhar(self, camera: Camera, fps: float, pausado: bool, t_slerp: float):
        """Desenha o painel de informações."""
        # ── Painel superior esquerdo ──────────────────────────────────────
        painel_x, painel_y = 12, 12
        linhas = [
            ("▶ VISUALIZADOR 3D", CIANO, self.fonte_titulo),
            (f"  FPS: {fps:5.1f}", VERDE_NEON, self.fonte_info),
            (f"  FOV: {camera.fov:5.1f}°", AMARELO, self.fonte_info),
            (f"  Câm: ({camera.posicao[0]:+.2f}, {camera.posicao[1]:+.2f}, {camera.posicao[2]:+.2f})", CINZA_TEXTO, self.fonte_info),
            (f"  Yaw: {math.degrees(camera.yaw):+.1f}°  Pitch: {math.degrees(camera.pitch):+.1f}°", CINZA_TEXTO, self.fonte_info),
        ]
        if pausado:
            linhas.append(("  ⏸ PAUSADO", LARANJA, self.fonte_titulo))

        y_atual = painel_y
        for texto, cor, fonte in linhas:
            self._texto(texto, painel_x, y_atual, cor, fonte)
            y_atual += fonte.get_height() + 3

        # ── Barra SLERP ───────────────────────────────────────────────────
        barra_x, barra_y = 12, self.h - 65
        self._texto("SLERP — Transição Suave entre Rotações:", barra_x, barra_y, CINZA_TEXTO, self.fonte_small)
        barra_largura = 220
        barra_altura = 10
        pygame.draw.rect(self.sup, (40, 40, 60), (barra_x, barra_y + 15, barra_largura, barra_altura), border_radius=5)
        fill_w = int(t_slerp * barra_largura)
        if fill_w > 0:
            pygame.draw.rect(self.sup, VIOLETA, (barra_x, barra_y + 15, fill_w, barra_altura), border_radius=5)
        pygame.draw.rect(self.sup, CINZA_GRADE, (barra_x, barra_y + 15, barra_largura, barra_altura), 1, border_radius=5)
        self._texto(f"t = {t_slerp:.2f}", barra_x + barra_largura + 8, barra_y + 14, VIOLETA, self.fonte_small)

        # ── Painel de controles (canto direito) ───────────────────────────
        controles = [
            "CONTROLES",
            "W/S  — frente / trás",
            "A/D  — esquerda / direita",
            "Q/E  — baixo / cima",
            "↑↓←→ — girar câmera",
            "Z/X  — zoom in / out",
            "SPC  — pausar",
            "R    — resetar câmera",
            "ESC  — sair",
        ]
        cx = self.w - 175
        cy = 12
        for i, linha in enumerate(controles):
            cor = CIANO if i == 0 else CINZA_TEXTO
            fonte = self.fonte_titulo if i == 0 else self.fonte_small
            self._texto(linha, cx, cy, cor, fonte)
            cy += fonte.get_height() + 4

        # ── Legenda dos eixos ─────────────────────────────────────────────
        leg_x, leg_y = self.w - 175, self.h - 75
        self._texto("EIXOS:", leg_x, leg_y, CINZA_TEXTO, self.fonte_small)
        self._texto("● X (vermelho)", leg_x, leg_y + 14, VERMELHO_GLOW, self.fonte_small)
        self._texto("● Y (verde)", leg_x, leg_y + 27, VERDE_NEON, self.fonte_small)
        self._texto("● Z (azul)", leg_x, leg_y + 40, (80, 140, 255), self.fonte_small)


# ════════════════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

def main():
    """
    Loop principal da aplicação de visualização 3D.

    Organização:
      1. Inicialização do Pygame e criação da janela
      2. Criação dos objetos 3D e câmera
      3. Loop principal:
         a. Processamento de eventos (teclado/mouse)
         b. Atualização da animação (ângulos, SLERP)
         c. Renderização (pipeline 3D → 2D)
         d. HUD overlay
         e. Display flip (double buffer)
    """

    # ── 1. Inicialização ──────────────────────────────────────────────────
    pygame.init()
    pygame.display.set_caption("Visualizador 3D — NumPy + Pygame | CG 2025")

    LARGURA, ALTURA = 1280, 720
    tela = pygame.display.set_mode((LARGURA, ALTURA), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    FPS_ALVO = 60

    # ── 2. Objetos 3D ─────────────────────────────────────────────────────
    vertices_cubo, arestas_cubo = criar_cubo(tamanho=1.0)
    vertices_piramide, arestas_piramide = criar_piramide(base=0.7, altura=1.4)
    vertices_grid, arestas_grid = criar_grid(tamanho=7, passo=1.0)
    eixos = criar_eixos(comprimento=1.5)

    # ── 3. Câmera e Renderizador ──────────────────────────────────────────
    camera = Camera(
        posicao=np.array([0.0, 3.0, 9.0]),
        fov=60.0,
        aspect_ratio=LARGURA / ALTURA
    )
    renderizador = Renderizador(tela)
    hud = HUD(tela)

    # ── 4. Estado da animação ─────────────────────────────────────────────
    tempo_total = 0.0           # segundos acumulados
    pausado = False

    # Quatérnios para SLERP Demo:
    # Transição suave entre dois estados de rotação do cubo
    q_slerp_inicio = Quaternion.from_axis_angle([1, 0, 0], 0)
    q_slerp_fim    = Quaternion.from_axis_angle([1, 1, 0], np.pi)
    t_slerp = 0.0               # parâmetro de interpolação [0→1→0]
    slerp_direcao = 1           # 1 = avançando, -1 = voltando

    # ── 5. Controle de mouse ──────────────────────────────────────────────
    mouse_capturado = False
    ultimo_mouse = (0, 0)

    # ── Loop principal ────────────────────────────────────────────────────
    rodando = True
    while rodando:

        dt = clock.tick(FPS_ALVO) / 1000.0  # delta time em segundos
        dt = min(dt, 0.05)                    # cap para evitar saltos

        if not pausado:
            tempo_total += dt

        # ── EVENTOS ───────────────────────────────────────────────────────
        for evento in pygame.event.get():

            if evento.type == pygame.QUIT:
                rodando = False

            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    rodando = False
                elif evento.key == pygame.K_SPACE:
                    pausado = not pausado
                elif evento.key == pygame.K_r:
                    # Reseta câmera
                    camera.posicao = np.array([0.0, 3.0, 9.0])
                    camera.yaw = -np.pi / 2
                    camera.pitch = -0.2
                    camera.fov = 60.0
                elif evento.key == pygame.K_m:
                    # Alterna captura do mouse
                    mouse_capturado = not mouse_capturado
                    pygame.mouse.set_visible(not mouse_capturado)

            elif evento.type == pygame.MOUSEBUTTONDOWN:
                if evento.button == 1:
                    mouse_capturado = True
                    pygame.mouse.set_visible(False)
                    ultimo_mouse = pygame.mouse.get_pos()

            elif evento.type == pygame.MOUSEBUTTONUP:
                if evento.button == 1:
                    mouse_capturado = False
                    pygame.mouse.set_visible(True)

            elif evento.type == pygame.MOUSEMOTION and mouse_capturado:
                mx, my = evento.pos
                dx = mx - ultimo_mouse[0]
                dy = my - ultimo_mouse[1]
                camera.rotacionar(dx * 0.005, -dy * 0.005)
                ultimo_mouse = (mx, my)

            elif evento.type == pygame.VIDEORESIZE:
                LARGURA, ALTURA = evento.w, evento.h
                tela = pygame.display.set_mode((LARGURA, ALTURA), pygame.RESIZABLE)
                camera.aspect_ratio = LARGURA / ALTURA
                renderizador.largura = LARGURA
                renderizador.altura = ALTURA
                hud = HUD(tela)

        # ── INPUT CONTÍNUO (teclado) ──────────────────────────────────────
        teclas = pygame.key.get_pressed()

        # Movimento da câmera
        vel = 4.0 * dt
        if teclas[pygame.K_w]:      camera.mover_frente(vel)
        if teclas[pygame.K_s]:      camera.mover_tras(vel)
        if teclas[pygame.K_a]:      camera.mover_esquerda(vel)
        if teclas[pygame.K_d]:      camera.mover_direita(vel)
        if teclas[pygame.K_q]:      camera.mover_baixo(vel)
        if teclas[pygame.K_e]:      camera.mover_cima(vel)

        # Rotação da câmera com setas
        rot_vel = 1.5 * dt
        if teclas[pygame.K_LEFT]:   camera.rotacionar(-rot_vel, 0)
        if teclas[pygame.K_RIGHT]:  camera.rotacionar( rot_vel, 0)
        if teclas[pygame.K_UP]:     camera.rotacionar(0,  rot_vel)
        if teclas[pygame.K_DOWN]:   camera.rotacionar(0, -rot_vel)

        # Zoom
        if teclas[pygame.K_z]:      camera.zoom(-30 * dt)
        if teclas[pygame.K_x]:      camera.zoom( 30 * dt)

        # ── ATUALIZAÇÃO DA ANIMAÇÃO ───────────────────────────────────────
        if not pausado:
            # SLERP: oscila t entre 0 e 1 suavemente
            t_slerp += slerp_direcao * dt * 0.4
            if t_slerp >= 1.0:
                t_slerp = 1.0
                slerp_direcao = -1
            elif t_slerp <= 0.0:
                t_slerp = 0.0
                slerp_direcao = 1

        # Quatérnio atual do cubo via SLERP
        q_cubo_slerp = slerp(q_slerp_inicio, q_slerp_fim, t_slerp)

        # Rotação contínua do cubo: combina SLERP + rotação Y contínua
        q_spin_y = Quaternion.from_axis_angle([0, 1, 0], tempo_total * 0.8)
        q_spin_x = Quaternion.from_axis_angle([1, 0, 0], tempo_total * 0.5)
        q_cubo_total = q_spin_y * q_spin_x * q_cubo_slerp

        # Matriz de rotação do cubo (combina quatérnio → matrix 4×4)
        mat_rot_cubo = q_cubo_total.to_matrix()

        # Matriz do cubo: escala → rotação → posição na origem
        mat_cubo = compor(
            translacao(0, 0.5, 0),   # eleva levemente o cubo
            mat_rot_cubo,
            escala(1.0, 1.0, 1.0)
        )

        # Pirâmide orbita o cubo em um círculo no plano XZ
        raio_orbita = 3.5
        velocidade_orbita = 0.7
        angulo_orbita = tempo_total * velocidade_orbita
        px = math.cos(angulo_orbita) * raio_orbita
        pz = math.sin(angulo_orbita) * raio_orbita

        # Pirâmide também gira ao redor do próprio eixo Y
        q_giro_piramide = Quaternion.from_axis_angle([0, 1, 0], -tempo_total * 1.5)
        mat_rot_piramide = q_giro_piramide.to_matrix()

        mat_piramide = compor(
            translacao(px, 0.5, pz),   # posição orbital
            mat_rot_piramide,            # rotação própria
        )

        # ── RENDERIZAÇÃO ──────────────────────────────────────────────────
        # Fundo com gradiente vertical (simulado com retângulos)
        tela.fill(AZUL_ESCURO)

        # Gradiente sutil no topo
        for i in range(min(200, ALTURA)):
            alpha = i / 200
            cor_grad = (
                int(AZUL_ESCURO[0] * (1 - alpha) + 15 * alpha),
                int(AZUL_ESCURO[1] * (1 - alpha) + 20 * alpha),
                int(AZUL_ESCURO[2] * (1 - alpha) + 50 * alpha),
            )
            pygame.draw.line(tela, cor_grad, (0, i), (LARGURA, i))

        # Obtém matrizes da câmera
        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix()
        mat_id = np.eye(4, dtype=float)  # Matriz identidade (sem transformação)

        # Renderiza grid do chão
        renderizador.superficie = tela
        renderizador.largura = LARGURA
        renderizador.altura = ALTURA
        renderizador.renderizar_objeto(
            vertices_grid, arestas_grid,
            mat_id, view, proj,
            cor=CINZA_GRADE
        )

        # Renderiza eixos de referência
        for v_eixo, a_eixo, cor_eixo in eixos:
            renderizador.renderizar_objeto(
                v_eixo, a_eixo,
                mat_id, view, proj,
                cor=cor_eixo
            )

        # Renderiza o CUBO com brilho duplo (glow effect)
        # Primeiro uma versão mais escura (bordas)
        renderizador.renderizar_objeto(
            vertices_cubo, arestas_cubo,
            mat_cubo, view, proj,
            cor=(0, 80, 120)
        )
        # Depois a versão brilhante (CIANO)
        renderizador.renderizar_objeto(
            vertices_cubo, arestas_cubo,
            mat_cubo, view, proj,
            cor=CIANO
        )

        # Renderiza a PIRÂMIDE orbitando
        renderizador.renderizar_objeto(
            vertices_piramide, arestas_piramide,
            mat_piramide, view, proj,
            cor=(200, 80, 255)  # Violeta
        )
        # Versão mais brilhante da pirâmide
        mat_piramide_glow = compor(
            translacao(px, 0.5, pz),
            mat_rot_piramide,
            escala(0.98, 0.98, 0.98)  # Levemente menor para efeito glow
        )
        renderizador.renderizar_objeto(
            vertices_piramide, arestas_piramide,
            mat_piramide_glow, view, proj,
            cor=LARANJA
        )

        # ── HUD Overlay ────────────────────────────────────────────────────
        fps = clock.get_fps()
        hud.desenhar(camera, fps, pausado, t_slerp)

        # ── Linha divisória decorativa ─────────────────────────────────────
        pygame.draw.line(tela, (30, 30, 60), (0, ALTURA - 80), (LARGURA, ALTURA - 80), 1)

        # Mensagem de controle de mouse
        msg = "  Clique e arraste para girar a câmera"
        fonte_m = pygame.font.SysFont("Consolas", 11)
        surf_m = fonte_m.render(msg, True, (60, 60, 90))
        tela.blit(surf_m, (LARGURA // 2 - surf_m.get_width() // 2, ALTURA - 75))

        # ── Apresenta o frame (double buffer) ─────────────────────────────
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
