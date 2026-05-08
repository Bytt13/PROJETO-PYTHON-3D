"""
=============================================================================
 PARTE 5: APLICAÇÃO PRINCIPAL — MENU + VISUALIZADOR 3D SÓLIDO
=============================================================================

 Controles no MENU:
   ↑ / ↓      → navegar entre os sólidos
   ENTER      → entrar no visualizador

 Controles no VISUALIZADOR (modo AUTO):
   W/S        → mover câmera frente/trás
   A/D        → mover câmera esquerda/direita
   Q/E        → mover câmera baixo/cima
   ↑/↓/←/→   → girar câmera
   Z/X        → zoom (ajustar FOV)
   R          → resetar câmera
   ESPAÇO     → pausar rotação automática
   TAB        → alternar para modo MANUAL
   ESC        → voltar ao menu

 Controles no VISUALIZADOR (modo MANUAL):
   TAB        → voltar ao modo AUTO
   1-7        → selecionar transformação ativa
   Mouse drag → ajustar parâmetros da transformação ativa
   Scroll     → ajuste fino (eixo Z / escala / ângulo)
   BACKSPACE  → resetar transformação selecionada
   Click      → selecionar card de transformação no painel
   Setas      → girar câmera (mantido)
   R          → resetar câmera
   ESC        → voltar ao menu

 Transformações disponíveis no modo MANUAL:
   1 Translação    — mouse X→TX, Y→TY, scroll→TZ
   2 Escala         — mouse Y→uniforme, scroll→uniforme
   3 Rotação X      — mouse Y→ângulo (Euler)
   4 Rotação Y      — mouse X→ângulo (Euler)
   5 Rotação Z      — mouse X→ângulo (Euler)
   6 SLERP          — mouse X→t (interpolação quaternion)
   7 Cisalhamento   — mouse X→a, Y→b (shear XY)

=============================================================================
"""

import sys
import math
import random
import numpy as np
import pygame

from engine.transforms import translacao, escala, rotacao_x, rotacao_y, rotacao_z, cisalhamento_xy, compor
from engine.quaternion import Quaternion, slerp as q_slerp
from engine.camera import Camera
from engine.renderer import Renderizador
from engine.mesh import FORMAS, Mesh, Objeto3D

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTES GLOBAIS
# ════════════════════════════════════════════════════════════════════════════
W, H = 1280, 720
FPS  = 60

# Paleta
C_BG        = (  6,   6,  16)
C_PANEL_DK  = ( 12,  12,  28)
C_PANEL_MD  = ( 18,  18,  40)
C_BORDER    = ( 35,  35,  75)
C_BORDER_HL = (  0, 210, 240)   # ciano — highlight
C_TEXT_DIM  = (100, 100, 140)
C_TEXT      = (185, 185, 215)
C_WHITE     = (240, 240, 255)
C_ACCENT    = (  0, 210, 240)
C_ACCENT2   = (170,  60, 255)

# Direção da luz (world space)
LUZ_DIR = np.array([0.6, 1.0, -0.8], dtype=float)
LUZ_DIR = LUZ_DIR / np.linalg.norm(LUZ_DIR)


# ════════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS DE UI
# ════════════════════════════════════════════════════════════════════════════

def lerp_color(a, b, t):
    return tuple(int(a[i] + (b[i]-a[i])*t) for i in range(3))

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

class FonteCache:
    """Cache de fontes pygame para não recriar a cada frame."""
    _cache = {}
    @staticmethod
    def get(nome, tamanho, bold=False):
        key = (nome, tamanho, bold)
        if key not in FonteCache._cache:
            FonteCache._cache[key] = pygame.font.SysFont(nome, tamanho, bold=bold)
        return FonteCache._cache[key]

def texto(surf, msg, x, y, cor=C_TEXT, nome="Consolas", tam=14, bold=False, centro=False):
    f = FonteCache.get(nome, tam, bold)
    s = f.render(str(msg), True, cor)
    if centro:
        x -= s.get_width() // 2
    surf.blit(s, (x, y))
    return s.get_width(), s.get_height()

def rect_aa(surf, cor, rx, ry, rw, rh, raio=8, borda=0, cor_borda=None):
    """Desenha retângulo com cantos arredondados."""
    pygame.draw.rect(surf, cor, (rx, ry, rw, rh), borda, border_radius=raio)
    if borda and cor_borda:
        pygame.draw.rect(surf, cor_borda, (rx, ry, rw, rh), borda, border_radius=raio)

def glow_line(surf, cor, p1, p2, largura=2):
    """Linha com 'glow' (camadas sobrepostas)."""
    pygame.draw.line(surf, tuple(c//4 for c in cor), p1, p2, largura+4)
    pygame.draw.line(surf, tuple(c//2 for c in cor), p1, p2, largura+2)
    pygame.draw.line(surf, cor, p1, p2, largura)


# ════════════════════════════════════════════════════════════════════════════
# SISTEMA DE PARTÍCULAS (fundo animado)
# ════════════════════════════════════════════════════════════════════════════

class Particle:
    def __init__(self, w, h):
        self.reset(w, h, aleatorio=True)

    def reset(self, w, h, aleatorio=False):
        self.x = random.uniform(0, w)
        self.y = random.uniform(0, h) if aleatorio else -5
        self.r = random.uniform(0.5, 2.0)
        self.vy = random.uniform(12, 40)
        self.vx = random.uniform(-8, 8)
        self.alpha = random.uniform(40, 120)
        self.cor = random.choice([
            (0, 150, 220), (130, 60, 220), (0, 200, 130), (200, 200, 255)
        ])

    def update(self, dt, w, h):
        self.x += self.vx * dt
        self.y += self.vy * dt
        if self.y > h + 10 or self.x < -10 or self.x > w + 10:
            self.reset(w, h)

    def draw(self, surf):
        ix, iy, ir = int(self.x), int(self.y), max(1, int(self.r))
        pygame.draw.circle(surf, self.cor, (ix, iy), ir)


def criar_particulas(n=70):
    return [Particle(W, H) for _ in range(n)]


# ════════════════════════════════════════════════════════════════════════════
# TELA DE MENU
# ════════════════════════════════════════════════════════════════════════════

class Menu:
    CARD_H   = 62
    CARD_GAP = 8
    LIST_X   = 60
    LIST_W   = 370
    DESC_X   = 480
    DESC_W   = W - 480 - 50

    def __init__(self, surf: pygame.Surface, particulas, larg=W, alt=H):
        self.surf       = surf
        self.particulas = particulas
        self.selecionado = 0
        # +1 para a opcao "Cena Demo" no topo do menu
        self.n = len(FORMAS) + 1
        self.larg = larg
        self.alt  = alt
        self._scroll_anim = 0.0
        self._pulse = 0.0

    def _card_y(self, idx):
        """Y do card para o índice idx."""
        top = self.alt // 2 - (self.n / 2) * (self.CARD_H + self.CARD_GAP)
        return int(top + idx * (self.CARD_H + self.CARD_GAP))

    def handle_event(self, ev):
        """Retorna 'viewer' se ENTER, None caso contrário."""
        if ev.type == pygame.KEYDOWN:
            if ev.key in (pygame.K_DOWN, pygame.K_s):
                self.selecionado = (self.selecionado + 1) % self.n
            elif ev.key in (pygame.K_UP, pygame.K_w):
                self.selecionado = (self.selecionado - 1) % self.n
            elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                if self.selecionado == 0:
                    return "demo"
                return "viewer"
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            mx, my = ev.pos
            for i in range(self.n):
                cy = self._card_y(i)
                if self.LIST_X <= mx <= self.LIST_X + self.LIST_W and cy <= my <= cy + self.CARD_H:
                    if self.selecionado == i:
                        if i == 0:
                            return "demo"
                        return "viewer"
                    self.selecionado = i
        return None

    def update(self, dt):
        self._pulse = (self._pulse + dt * 2.0) % (2 * math.pi)

    def draw(self, larg=W, alt=H):
        self.larg, self.alt = larg, alt
        surf = self.surf
        surf.fill(C_BG)

        # ── Fundo gradiente sutil ────────────────────────────────────────────
        for i in range(0, alt, 3):
            t = i / alt
            r = int(6 + 12*t); g = int(6 + 6*t); b = int(16 + 24*t)
            pygame.draw.line(surf, (r,g,b), (0,i), (larg,i))
            pygame.draw.line(surf, (r,g,b), (0,i+1), (larg,i+1))
            pygame.draw.line(surf, (r,g,b), (0,i+2), (larg,i+2))

        # ── Partículas ─────────────────────────────────────────────────────
        for p in self.particulas:
            p.draw(surf)

        # ── Título ─────────────────────────────────────────────────────────────────
        pulse = (math.sin(self._pulse) + 1) / 2
        cor_titulo = lerp_color(C_ACCENT, C_ACCENT2, pulse)
        texto(surf, "VISUALIZADOR 3D", larg//2, 28, cor_titulo, "Consolas", 40, bold=True, centro=True)
        texto(surf, "Computação Gráfica — Sólidos Geométricos",
              larg//2, 80, C_TEXT_DIM, "Consolas", 15, centro=True)
        glow_line(surf, C_ACCENT, (larg//4, 105), (3*larg//4, 105), 1)

        # ── Lista de opcoes (esquerda) ─────────────────────────────────────
        # Item 0 = Cena Demo, itens 1..N = solidos
        all_items = self.n  # total de itens no menu
        for i in range(all_items):
            cy = self._card_y(i)
            selecionado = (i == self.selecionado)

            if i == 0:
                # ── Card especial: Cena Demo (Parte 5) ─────────────────────
                card_cor = lerp_color(C_PANEL_MD, C_ACCENT, 0.15) if selecionado else C_PANEL_DK
                rect_aa(surf, card_cor, self.LIST_X, cy, self.LIST_W, self.CARD_H, raio=12)
                bord_cor = cor_titulo if selecionado else C_BORDER
                rect_aa(surf, (0,0,0,0), self.LIST_X, cy, self.LIST_W, self.CARD_H,
                        raio=12, borda=2 if selecionado else 1, cor_borda=bord_cor)

                nome_cor = C_WHITE if selecionado else C_TEXT
                texto(surf, "*", self.LIST_X + 16, cy + 10, C_ACCENT, "Consolas", 12)
                texto(surf, "Cena Animada (Parte 5)", self.LIST_X + 50, cy + 12,
                      nome_cor, "Consolas", 17, bold=selecionado)
                texto(surf, "Cubo + Piramide orbitando + SLERP",
                      self.LIST_X + 50, cy + 36, C_TEXT_DIM, "Consolas", 11)
                if selecionado:
                    texto(surf, ">", self.LIST_X + self.LIST_W - 28, cy + 21,
                          cor_titulo, "Consolas", 16)
                continue

            # ── Cards de solidos (itens 1..N) ──────────────────────────────
            forma = FORMAS[i - 1]  # offset -1 por causa do item demo

            # Fundo do card
            card_cor = C_PANEL_MD if selecionado else C_PANEL_DK
            rect_aa(surf, card_cor, self.LIST_X, cy, self.LIST_W, self.CARD_H, raio=10)

            # Borda
            borda_cor = cor_titulo if selecionado else C_BORDER
            borda_w = 2 if selecionado else 1
            rect_aa(surf, (0,0,0,0), self.LIST_X, cy, self.LIST_W, self.CARD_H,
                    raio=10, borda=borda_w, cor_borda=borda_cor)

            # Cor lateral (indicador)
            if selecionado:
                pygame.draw.rect(surf, forma.cor,
                                 (self.LIST_X, cy+10, 4, self.CARD_H-20), border_radius=2)

            # Número + Ícone + Nome
            num_cor  = forma.cor if selecionado else C_TEXT_DIM
            nome_cor = C_WHITE   if selecionado else C_TEXT

            texto(surf, f"{i+1:02d}", self.LIST_X + 16, cy + 10, num_cor, "Consolas", 12)
            texto(surf, forma.nome, self.LIST_X + 50, cy + 12,
                  nome_cor, "Consolas", 17, bold=selecionado)

            # Propriedades em miniatura
            props = forma.propriedades
            mini = f"V:{props.get('Vértices','?')}  F:{props.get('Faces','?')}  A:{props.get('Arestas','?')}"
            texto(surf, mini, self.LIST_X + 50, cy + 36, C_TEXT_DIM, "Consolas", 11)

            # Seta indicadora
            if selecionado:
                texto(surf, ">", self.LIST_X + self.LIST_W - 28, cy + 21,
                      cor_titulo, "Consolas", 16)

        # ── Painel de descrição (direita) ──────────────────────────────────────────
        dx, dy = self.DESC_X, 120
        dw = larg - self.DESC_X - 50
        dh = alt - 180

        rect_aa(surf, C_PANEL_DK, dx, dy-10, dw, dh, raio=14)
        rect_aa(surf, (0,0,0,0), dx, dy-10, dw, dh, raio=14, borda=1, cor_borda=C_BORDER)

        if self.selecionado == 0:
            # ── Descricao da Cena Demo ──────────────────────────────────────
            pygame.draw.rect(surf, C_ACCENT, (dx, dy-10, dw, 5), border_radius=14)
            texto(surf, "Cena Animada", dx + dw//2, dy + 10,
                  C_ACCENT, "Consolas", 30, bold=True, centro=True)

            tw, _ = texto(surf, "Parte 5 (0,5 extra)", dx + dw//2, dy + 52,
                          C_TEXT_DIM, "Consolas", 13, centro=True)
            rect_aa(surf, C_PANEL_MD, dx + dw//2 - tw//2 - 10, dy + 48,
                    tw + 20, 20, raio=10)
            texto(surf, "Parte 5 (0,5 extra)", dx + dw//2, dy + 52,
                  C_TEXT_DIM, "Consolas", 13, centro=True)

            pygame.draw.line(surf, C_BORDER, (dx+20, dy+78), (dx+dw-20, dy+78))

            desc_lines = [
                "Cubo rotacionando via quaternios",
                "Piramide orbitando em torno do cubo",
                "Controles de camera (WASD + mouse)",
                "Zoom interativo (Z/X ou FOV)",
                "SLERP entre duas rotacoes",
            ]
            ry = dy + 95
            for line in desc_lines:
                texto(surf, "- " + line, dx + 30, ry, C_TEXT, "Consolas", 13)
                ry += 22

        else:
            # ── Descricao do solido selecionado ─────────────────────────────
            forma = FORMAS[self.selecionado - 1]

            pygame.draw.rect(surf, forma.cor,
                             (dx, dy-10, dw, 5), border_radius=14)
            texto(surf, forma.nome, dx + dw//2, dy + 10,
                  forma.cor, "Consolas", 30, bold=True, centro=True)

            tipo = forma.propriedades.get("Tipo", "")
            tw, _ = texto(surf, tipo, dx + dw//2, dy + 52,
                          C_TEXT_DIM, "Consolas", 13, centro=True)
            rect_aa(surf, C_PANEL_MD, dx + dw//2 - tw//2 - 10, dy + 48,
                    tw + 20, 20, raio=10)
            texto(surf, tipo, dx + dw//2, dy + 52,
                  C_TEXT_DIM, "Consolas", 13, centro=True)

            pygame.draw.line(surf, C_BORDER,
                             (dx+20, dy+78), (dx+dw-20, dy+78))

            ry = dy + 92
            for k, v in forma.propriedades.items():
                if k == "Tipo":
                    continue
                texto(surf, k, dx + 30, ry, C_TEXT_DIM, "Consolas", 13)
                fw, _ = texto(surf, str(v), dx, ry, C_WHITE, "Consolas", 13)
                texto(surf, str(v), dx + dw - 30 - fw, ry, C_WHITE, "Consolas", 13)
                ry += 22

            pygame.draw.line(surf, C_BORDER,
                             (dx+20, ry+4), (dx+dw-20, ry+4))

            ry += 16
            for linha in forma.descricao.split("\n"):
                texto(surf, linha, dx + 30, ry, C_TEXT, "Consolas", 13)
                ry += 20

        # Botao ENTER
        btn_y = dy + dh - 52
        pulse2 = (math.sin(self._pulse * 1.5) + 1) / 2
        sel_cor = C_ACCENT if self.selecionado == 0 else FORMAS[self.selecionado - 1].cor
        btn_cor = lerp_color(C_PANEL_MD, sel_cor, pulse2 * 0.3)
        rect_aa(surf, btn_cor, dx + 40, btn_y, dw - 80, 36, raio=10)
        rect_aa(surf, (0,0,0), dx + 40, btn_y, dw - 80, 36,
                raio=10, borda=2, cor_borda=lerp_color(C_BORDER, sel_cor, pulse2))
        texto(surf, ">  ENTER   para visualizar",
              dx + dw//2, btn_y + 9, C_WHITE, "Consolas", 15, bold=True, centro=True)

        # ── Rodápé ─────────────────────────────────────────────────────────────────
        pygame.draw.line(surf, C_BORDER, (0, alt-36), (larg, alt-36))
        texto(surf, "W/S  Navegar    ENTER  Visualizar    ESC  Sair",
              larg//2, alt-26, C_TEXT_DIM, "Consolas", 12, centro=True)


# ════════════════════════════════════════════════════════════════════════════
# PARTE 5 — CENA DEMO: CUBO + PIRÂMIDE ORBITANDO + SLERP
# ════════════════════════════════════════════════════════════════════════════

class CenaDemo:
    """
    Cena completa conforme Parte 5 do enunciado (0,5 ponto extra):
      - Cubo rotacionando continuamente usando quaternios
      - Piramide orbitando em torno do cubo
      - Controles interativos (teclado/mouse) para mover, rotacionar e zoom
      - SLERP (interpolacao suave entre duas rotacoes)
    """

    def __init__(self, surf: pygame.Surface, larg=W, alt=H):
        self.surf = surf
        self.larg = larg
        self.alt  = alt
        self.renderer = Renderizador(surf)
        self.renderer.largura = larg
        self.renderer.altura  = alt

        # Camera olhando para a cena
        self.camera = Camera(
            posicao=np.array([0.0, 3.0, 10.0]),
            fov=55.0,
            aspect_ratio=larg/alt,
        )
        self.camera.yaw   = -math.pi / 2
        self.camera.pitch = -0.25

        # Meshes da cena
        from engine.mesh import criar_cubo, criar_piramide
        self._cubo     = criar_cubo(1.0)
        self._piramide = criar_piramide()

        # ── Cubo: rotacao continua via quaternios ────────────────────────────
        self._q_cubo_rot  = Quaternion(1, 0, 0, 0)
        self._q_cubo_spin = Quaternion.from_axis_angle([0, 1, 0], 0.012)

        # ── Piramide: orbita + rotacao propria ───────────────────────────────
        self._orbit_angle = 0.0
        self._orbit_radius = 3.5
        self._orbit_speed  = 0.6   # rad/s
        self._q_pir_rot  = Quaternion(1, 0, 0, 0)
        self._q_pir_spin = Quaternion.from_axis_angle([0, 1, 0], -0.018)

        # ── SLERP demo ──────────────────────────────────────────────────────
        self._q_slerp_A   = Quaternion.from_axis_angle([1, 0, 0], 0)
        self._q_slerp_B   = Quaternion.from_axis_angle([0, 1, 1], math.pi * 0.9)
        self._t_slerp     = 0.0
        self._slerp_dir   = 1

        self._pausado = False
        self._tempo   = 0.0
        self._mouse_drag  = False
        self._ultimo_mouse = (0, 0)

    def handle_event(self, ev):
        """Retorna 'menu' se ESC, None caso contrario."""
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                return "menu"
            if ev.key == pygame.K_SPACE:
                self._pausado = not self._pausado
            if ev.key == pygame.K_r:
                self.camera.posicao = np.array([0.0, 3.0, 10.0])
                self.camera.yaw   = -math.pi / 2
                self.camera.pitch = -0.25
                self.camera.fov   = 55.0
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self._mouse_drag = True
            self._ultimo_mouse = ev.pos
        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self._mouse_drag = False
        if ev.type == pygame.MOUSEMOTION and self._mouse_drag:
            dx = ev.pos[0] - self._ultimo_mouse[0]
            dy = ev.pos[1] - self._ultimo_mouse[1]
            self.camera.rotacionar(dx * 0.005, -dy * 0.005)
            self._ultimo_mouse = ev.pos
        return None

    def update(self, dt):
        if not self._pausado:
            self._tempo += dt

            # Cubo: rotacao automatica via produto de quaternios
            self._q_cubo_rot = self._q_cubo_spin * self._q_cubo_rot

            # Piramide: atualiza angulo de orbita + rotacao propria
            self._orbit_angle += self._orbit_speed * dt
            self._q_pir_rot = self._q_pir_spin * self._q_pir_rot

            # SLERP: oscila t entre 0 e 1
            self._t_slerp += self._slerp_dir * dt * 0.35
            if self._t_slerp >= 1.0:
                self._t_slerp = 1.0
                self._slerp_dir = -1
            elif self._t_slerp <= 0.0:
                self._t_slerp = 0.0
                self._slerp_dir = 1

        # Input continuo de camera (translacao, rotacao, zoom)
        teclas = pygame.key.get_pressed()
        vel = 4.0 * dt
        if teclas[pygame.K_w]:      self.camera.mover_frente(vel)
        if teclas[pygame.K_s]:      self.camera.mover_tras(vel)
        if teclas[pygame.K_a]:      self.camera.mover_esquerda(vel)
        if teclas[pygame.K_d]:      self.camera.mover_direita(vel)
        if teclas[pygame.K_q]:      self.camera.mover_baixo(vel)
        if teclas[pygame.K_e]:      self.camera.mover_cima(vel)
        rv = 1.8 * dt
        if teclas[pygame.K_LEFT]:   self.camera.rotacionar(-rv, 0)
        if teclas[pygame.K_RIGHT]:  self.camera.rotacionar( rv, 0)
        if teclas[pygame.K_UP]:     self.camera.rotacionar(0,  rv)
        if teclas[pygame.K_DOWN]:   self.camera.rotacionar(0, -rv)
        if teclas[pygame.K_z]:      self.camera.zoom(-28*dt)
        if teclas[pygame.K_x]:      self.camera.zoom( 28*dt)

    def draw(self, fps: float):
        surf = self.surf

        # ── Fundo gradiente ────────────────────────────────────────────────
        surf.fill(C_BG)
        for i in range(0, self.alt, 4):
            t = i / self.alt
            r = int(6 + 10*t); g = int(6 + 5*t); b = int(16 + 20*t)
            pygame.draw.line(surf, (r,g,b), (0,i), (self.larg,i))
            pygame.draw.line(surf, (r,g,b), (0,i+1), (self.larg,i+1))
            pygame.draw.line(surf, (r,g,b), (0,i+2), (self.larg,i+2))
            pygame.draw.line(surf, (r,g,b), (0,i+3), (self.larg,i+3))

        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix()
        self.renderer.superficie = surf
        self.renderer.largura    = self.larg
        self.renderer.altura     = self.alt

        # ── Grid de chao ────────────────────────────────────────────────────
        grid_y = -0.3
        for g in range(-6, 7):
            cor_grid = (20, 20, 50)
            p_a = np.array([g, grid_y, -6, 1.0])
            p_b = np.array([g, grid_y,  6, 1.0])
            p_c = np.array([-6, grid_y, g, 1.0])
            p_d = np.array([ 6, grid_y, g, 1.0])
            mvp = proj @ view
            ca = mvp @ p_a; cb = mvp @ p_b
            cc = mvp @ p_c; cd = mvp @ p_d
            s1 = self.renderer._clip_to_ndc_screen(ca)
            s2 = self.renderer._clip_to_ndc_screen(cb)
            s3 = self.renderer._clip_to_ndc_screen(cc)
            s4 = self.renderer._clip_to_ndc_screen(cd)
            if s1 and s2:
                pygame.draw.aaline(surf, cor_grid, s1, s2)
            if s3 and s4:
                pygame.draw.aaline(surf, cor_grid, s3, s4)

        # ── CUBO: rotacao continua via quaternios ──────────────────────────
        q_slerp_atual = q_slerp(self._q_slerp_A, self._q_slerp_B, self._t_slerp)
        q_cubo_total = self._q_cubo_rot * q_slerp_atual
        mat_cubo = q_cubo_total.to_matrix()
        # Cubo fica no centro, escala 1
        mat_cubo_modelo = mat_cubo

        self.renderer.renderizar_solido(
            self._cubo.vertices,
            self._cubo.faces,
            mat_cubo_modelo,
            view, proj,
            cor_base=self._cubo.cor,
            camera_pos=self.camera.posicao,
            luz_dir=LUZ_DIR,
            bordas=True,
        )

        # ── PIRAMIDE: orbitando em torno do cubo ───────────────────────────
        # Posicao orbital: translacao circular no plano XZ
        ox = self._orbit_radius * math.cos(self._orbit_angle)
        oz = self._orbit_radius * math.sin(self._orbit_angle)
        oy = 0.4 + 0.3 * math.sin(self._orbit_angle * 2)  # leve flutuacao vertical

        mat_pir_rot = self._q_pir_rot.to_matrix()
        mat_pir_escala = escala(0.6, 0.6, 0.6)
        mat_pir_trans = translacao(ox, oy, oz)
        # Composicao: primeiro escala, depois rota, depois translada
        mat_pir_modelo = compor(mat_pir_trans, mat_pir_rot, mat_pir_escala)

        self.renderer.renderizar_solido(
            self._piramide.vertices,
            self._piramide.faces,
            mat_pir_modelo,
            view, proj,
            cor_base=self._piramide.cor,
            camera_pos=self.camera.posicao,
            luz_dir=LUZ_DIR,
            bordas=True,
        )

        # ── HUD ────────────────────────────────────────────────────────────
        self._draw_hud(fps)

    def _draw_hud(self, fps):
        surf = self.surf
        cam = self.camera

        # Titulo da cena
        rect_aa(surf, C_PANEL_DK, 10, 10, 280, 28, raio=8)
        rect_aa(surf, (0,0,0,0), 10, 10, 280, 28, raio=8, borda=1, cor_borda=C_BORDER)
        texto(surf, "PARTE 5: Cena Animada + SLERP", 20, 15, C_ACCENT, "Consolas", 13, bold=True)

        # Info da cena
        rect_aa(surf, C_PANEL_DK, 10, 44, 280, 68, raio=8)
        rect_aa(surf, (0,0,0,0), 10, 44, 280, 68, raio=8, borda=1, cor_borda=C_BORDER)
        texto(surf, "Cubo: rotacao por quaternios", 20, 50, C_ACCENT2, "Consolas", 11)
        texto(surf, "Piramide: orbita circular (translacao)", 20, 65, (255,205,30), "Consolas", 11)
        texto(surf, f"Orbita: {math.degrees(self._orbit_angle) % 360:.0f} graus", 20, 80, C_TEXT_DIM, "Consolas", 11)
        texto(surf, f"FPS: {fps:.0f}", 20, 95, C_TEXT_DIM, "Consolas", 11)

        # ── Barra SLERP ───────────────────────────────────────────────────
        rect_aa(surf, C_PANEL_DK, 10, self.alt-64, 265, 52, raio=10)
        rect_aa(surf, (0,0,0,0),  10, self.alt-64, 265, 52, raio=10, borda=1, cor_borda=C_BORDER)
        texto(surf, "SLERP  (interpolacao suave)", 18, self.alt-58, C_TEXT_DIM, "Consolas", 11)

        bw = 230
        pygame.draw.rect(surf, (30,30,60), (18, self.alt-42, bw, 10), border_radius=5)
        fill_w = int(self._t_slerp * bw)
        if fill_w > 0:
            pygame.draw.rect(surf, C_ACCENT2, (18, self.alt-42, fill_w, 10), border_radius=5)
        pygame.draw.rect(surf, C_BORDER, (18, self.alt-42, bw, 10), 1, border_radius=5)
        texto(surf, f"t={self._t_slerp:.2f}", 258, self.alt-44, C_ACCENT2, "Consolas", 11)

        # ── Controles ────────────────────────────────────────────────────
        cx = self.larg - 200
        ctrl = [
            ("CONTROLES",  C_ACCENT,   True,  14),
            ("W/S",        C_TEXT_DIM, False, 12),
            ("A/D",        C_TEXT_DIM, False, 12),
            ("Q/E",        C_TEXT_DIM, False, 12),
            ("Setas",      C_TEXT_DIM, False, 12),
            ("Z/X",        C_TEXT_DIM, False, 12),
            ("R",          C_TEXT_DIM, False, 12),
            ("ESPACO",     C_TEXT_DIM, False, 12),
            ("ESC",        C_TEXT_DIM, False, 12),
        ]
        vals = [
            ("",              C_ACCENT,   True,  14),
            ("frente / tras", C_TEXT,     False, 12),
            ("esq / dir",     C_TEXT,     False, 12),
            ("baixo / cima",  C_TEXT,     False, 12),
            ("girar camera",  C_TEXT,     False, 12),
            ("zoom",          C_TEXT,     False, 12),
            ("resetar",       C_TEXT,     False, 12),
            ("pausar",        C_TEXT,     False, 12),
            ("menu",          C_TEXT,     False, 12),
        ]
        rect_aa(surf, C_PANEL_DK, cx-10, 10, 200, 210, raio=10)
        rect_aa(surf, (0,0,0,0),  cx-10, 10, 200, 210, raio=10, borda=1, cor_borda=C_BORDER)

        cy2 = 20
        for (k, kc, kb, ks), (v, vc, vb, vs) in zip(ctrl, vals):
            texto(surf, k, cx+2,  cy2, kc, "Consolas", ks, bold=kb)
            texto(surf, v, cx+72, cy2, vc, "Consolas", vs)
            cy2 += ks + 5

        # ── Camera info ──────────────────────────────────────────────────
        cy3 = 230
        rect_aa(surf, C_PANEL_DK, cx-10, cy3, 200, 75, raio=10)
        rect_aa(surf, (0,0,0,0),  cx-10, cy3, 200, 75, raio=10, borda=1, cor_borda=C_BORDER)
        texto(surf, "CAMERA", cx, cy3+8, C_ACCENT, "Consolas", 12, bold=True)
        px, py2, pz = cam.posicao
        texto(surf, f"Pos  ({px:+.1f},{py2:+.1f},{pz:+.1f})", cx, cy3+26, C_TEXT_DIM, "Consolas", 10)
        texto(surf, f"FOV  {cam.fov:.0f}", cx, cy3+42, C_TEXT_DIM, "Consolas", 10)
        texto(surf, f"Yaw  {math.degrees(cam.yaw):.0f}   Pitch {math.degrees(cam.pitch):.0f}",
              cx, cy3+56, C_TEXT_DIM, "Consolas", 10)

        # ── Badge ESC ────────────────────────────────────────────────────
        rect_aa(surf, C_PANEL_MD, self.larg//2-80, self.alt-36, 160, 26, raio=8)
        texto(surf, "ESC -> voltar ao menu",
              self.larg//2, self.alt-29, C_TEXT_DIM, "Consolas", 12, centro=True)

        if self._pausado:
            texto(surf, "||  PAUSADO", self.larg//2, self.alt//2 - 20,
                  C_ACCENT, "Consolas", 22, bold=True, centro=True)


# ════════════════════════════════════════════════════════════════════════════
# TELA DO VISUALIZADOR 3D
# ════════════════════════════════════════════════════════════════════════════

class Visualizador:
    def __init__(self, surf: pygame.Surface, forma: Mesh, larg=W, alt=H):
        self.surf    = surf
        self.forma   = forma
        self.larg    = larg
        self.alt     = alt
        self.renderer = Renderizador(surf)
        self.renderer.largura = larg
        self.renderer.altura  = alt

        dist = max(5.0, float(np.max(np.abs(forma.vertices))) * 4.5)
        self.camera = Camera(
            posicao=np.array([0.0, 1.8, dist]),
            fov=55.0,
            aspect_ratio=larg/alt,
        )
        self.camera.yaw   = -math.pi / 2
        self.camera.pitch = -0.22

        # Quaterniões para rotação automática
        self._q_rot  = Quaternion(1, 0, 0, 0)
        self._q_spin = Quaternion.from_axis_angle([0,1,0], 0.009)

        # SLERP demo
        self._q_A     = Quaternion.from_axis_angle([1,0,0], 0)
        self._q_B     = Quaternion.from_axis_angle([0,1,1], math.pi * 0.9)
        self._t_slerp = 0.0
        self._sdir    = 1

        self._pausado = False
        self._tempo   = 0.0
        self._mouse_drag = False
        self._ultimo_mouse = (0, 0)

        # ── Modo manual interativo ──────────────────────────────────────
        self._modo_manual    = False
        self._transf_ativa   = 1        # 1..7
        self._manual_tx      = 0.0
        self._manual_ty      = 0.0
        self._manual_tz      = 0.0
        self._manual_esc     = 1.0      # escala uniforme
        self._manual_rx      = 0.0      # rotação Euler X (rad)
        self._manual_ry      = 0.0      # rotação Euler Y (rad)
        self._manual_rz      = 0.0      # rotação Euler Z (rad)
        self._manual_slerp_t = 0.0
        self._manual_shear_a = 0.0
        self._manual_shear_b = 0.0
        self._hud_cards      = []       # [(pygame.Rect, id), ...]

    def handle_event(self, ev):
        """Retorna 'menu' se ESC, None caso contrário."""
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                return "menu"
            if ev.key == pygame.K_SPACE:
                self._pausado = not self._pausado
            if ev.key == pygame.K_r:
                dist = max(5.0, float(np.max(np.abs(self.forma.vertices))) * 4.5)
                self.camera.posicao = np.array([0.0, 1.8, dist])
                self.camera.yaw   = -math.pi / 2
                self.camera.pitch = -0.22
                self.camera.fov   = 55.0
            # Toggle modo manual
            if ev.key == pygame.K_TAB:
                self._modo_manual = not self._modo_manual
            # Selecionar transformação ativa (1-7)
            if self._modo_manual:
                _NUM_KEYS = [pygame.K_1, pygame.K_2, pygame.K_3,
                             pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7]
                for i, k in enumerate(_NUM_KEYS, start=1):
                    if ev.key == k:
                        self._transf_ativa = i
                if ev.key == pygame.K_BACKSPACE:
                    self._resetar_transf_ativa()

        # Mouse click — selecionar card no HUD ou iniciar drag
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            clicou_card = False
            if self._modo_manual:
                mx, my = ev.pos
                for rect, tid in self._hud_cards:
                    if rect.collidepoint(mx, my):
                        self._transf_ativa = tid
                        clicou_card = True
                        break
            if not clicou_card:
                self._mouse_drag = True
                self._ultimo_mouse = ev.pos

        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self._mouse_drag = False

        if ev.type == pygame.MOUSEMOTION and self._mouse_drag:
            dx = ev.pos[0] - self._ultimo_mouse[0]
            dy = ev.pos[1] - self._ultimo_mouse[1]
            if self._modo_manual:
                self._aplicar_mouse_manual(dx, dy)
            else:
                self.camera.rotacionar(dx * 0.005, -dy * 0.005)
            self._ultimo_mouse = ev.pos

        # Scroll wheel — ajuste fino no modo manual
        if hasattr(pygame, 'MOUSEWHEEL') and ev.type == pygame.MOUSEWHEEL:
            if self._modo_manual:
                self._aplicar_scroll_manual(ev.y)

        return None

    # ── Helpers do modo manual ──────────────────────────────────────────

    def _resetar_transf_ativa(self):
        """Reseta a transformação ativa para valores padrão."""
        t = self._transf_ativa
        if t == 1:   self._manual_tx = self._manual_ty = self._manual_tz = 0.0
        elif t == 2: self._manual_esc = 1.0
        elif t == 3: self._manual_rx = 0.0
        elif t == 4: self._manual_ry = 0.0
        elif t == 5: self._manual_rz = 0.0
        elif t == 6: self._manual_slerp_t = 0.0
        elif t == 7: self._manual_shear_a = self._manual_shear_b = 0.0

    def _aplicar_mouse_manual(self, dx, dy):
        """Aplica movimento do mouse à transformação ativa."""
        t = self._transf_ativa
        s = 0.01                        # sensibilidade base
        if t == 1:                      # Translação
            self._manual_tx += dx * s
            self._manual_ty -= dy * s
        elif t == 2:                    # Escala uniforme
            self._manual_esc = max(0.1, self._manual_esc - dy * s)
        elif t == 3:                    # Rotação X
            self._manual_rx += dy * s * 2
        elif t == 4:                    # Rotação Y
            self._manual_ry += dx * s * 2
        elif t == 5:                    # Rotação Z
            self._manual_rz += dx * s * 2
        elif t == 6:                    # SLERP
            self._manual_slerp_t = clamp(self._manual_slerp_t + dx * s, 0.0, 1.0)
        elif t == 7:                    # Cisalhamento
            self._manual_shear_a += dx * s
            self._manual_shear_b -= dy * s

    def _aplicar_scroll_manual(self, direcao):
        """Aplica scroll do mouse à transformação ativa."""
        t = self._transf_ativa
        p = 0.1                         # passo por tick
        if t == 1:   self._manual_tz += direcao * p
        elif t == 2: self._manual_esc = max(0.1, self._manual_esc + direcao * p)
        elif t == 3: self._manual_rx += direcao * p * 0.5
        elif t == 4: self._manual_ry += direcao * p * 0.5
        elif t == 5: self._manual_rz += direcao * p * 0.5
        elif t == 6: self._manual_slerp_t = clamp(self._manual_slerp_t + direcao * 0.05, 0.0, 1.0)
        elif t == 7: self._manual_shear_a += direcao * p * 0.2

    def update(self, dt):
        if not self._pausado and not self._modo_manual:
            self._tempo   += dt
            # Rotação automática do sólido via produto de quatérnios
            self._q_rot    = self._q_spin * self._q_rot

            # Oscila t_slerp para demonstrar transição suave
            self._t_slerp += self._sdir * dt * 0.35
            if self._t_slerp >= 1.0:
                self._t_slerp = 1.0
                self._sdir = -1
            elif self._t_slerp <= 0.0:
                self._t_slerp = 0.0
                self._sdir = 1

        # Input contínuo de câmera
        teclas = pygame.key.get_pressed()
        vel = 4.0 * dt
        if teclas[pygame.K_w]:      self.camera.mover_frente(vel)
        if teclas[pygame.K_s]:      self.camera.mover_tras(vel)
        if teclas[pygame.K_a]:      self.camera.mover_esquerda(vel)
        if teclas[pygame.K_d]:      self.camera.mover_direita(vel)
        if teclas[pygame.K_q]:      self.camera.mover_baixo(vel)
        if teclas[pygame.K_e]:      self.camera.mover_cima(vel)
        rv = 1.8 * dt
        if teclas[pygame.K_LEFT]:   self.camera.rotacionar(-rv, 0)
        if teclas[pygame.K_RIGHT]:  self.camera.rotacionar( rv, 0)
        if teclas[pygame.K_UP]:     self.camera.rotacionar(0,  rv)
        if teclas[pygame.K_DOWN]:   self.camera.rotacionar(0, -rv)
        if teclas[pygame.K_z]:      self.camera.zoom(-28*dt)
        if teclas[pygame.K_x]:      self.camera.zoom( 28*dt)

    def draw(self, fps: float):
        surf = self.surf
        forma = self.forma

        # ── Fundo ──────────────────────────────────────────────────────────
        surf.fill(C_BG)
        for i in range(0, self.alt, 4):
            t = i / self.alt
            r = int(6  + 10*t); g = int(6  +  5*t); b = int(16 + 20*t)
            pygame.draw.line(surf, (r,g,b), (0,i),   (self.larg,i))
            pygame.draw.line(surf, (r,g,b), (0,i+1), (self.larg,i+1))
            pygame.draw.line(surf, (r,g,b), (0,i+2), (self.larg,i+2))
            pygame.draw.line(surf, (r,g,b), (0,i+3), (self.larg,i+3))

        # ── Matriz de modelo ────────────────────────────────────────────────
        escala_auto = 1.3 / max(0.01, float(np.max(np.abs(forma.vertices))))
        mat_esc_base = escala(escala_auto, escala_auto, escala_auto)

        if self._modo_manual:
            # Composição manual de todas as transformações
            m_trans  = translacao(self._manual_tx, self._manual_ty, self._manual_tz)
            m_esc    = escala(self._manual_esc, self._manual_esc, self._manual_esc)
            m_rx     = rotacao_x(self._manual_rx)
            m_ry     = rotacao_y(self._manual_ry)
            m_rz     = rotacao_z(self._manual_rz)
            m_shear  = cisalhamento_xy(self._manual_shear_a, self._manual_shear_b)
            q_sl     = q_slerp(self._q_A, self._q_B, self._manual_slerp_t)
            m_slerp  = q_sl.to_matrix()
            # Ordem: base → shear → escala → rotações Euler → slerp → translação
            mat_modelo = compor(m_trans, m_slerp, m_rz, m_ry, m_rx, m_esc, m_shear, mat_esc_base)
        else:
            # Modo automático (original)
            q_slerp_atual = q_slerp(self._q_A, self._q_B, self._t_slerp)
            q_total = self._q_rot * q_slerp_atual
            mat_rot = q_total.to_matrix()
            mat_modelo = compor(mat_rot, mat_esc_base)

        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix()

        # ── Grid de chão ───────────────────────────────────────────────────
        self._desenhar_grid(view, proj)

        # ── Sólido ─────────────────────────────────────────────────────────
        self.renderer.superficie = surf
        self.renderer.largura    = self.larg
        self.renderer.altura     = self.alt
        self.renderer.renderizar_solido(
            forma.vertices,
            forma.faces,
            mat_modelo,
            view,
            proj,
            cor_base=forma.cor,
            camera_pos=self.camera.posicao,
            luz_dir=LUZ_DIR,
            bordas=True,
        )

        # ── HUD ────────────────────────────────────────────────────────────
        self._draw_hud(fps)

    def _desenhar_grid(self, view, proj):
        """Grid XZ simples como chão de referência."""
        mat_id = np.eye(4, dtype=float)
        ren = self.renderer
        T = 5
        cor_g = (28, 28, 55)
        for i in range(-T, T+1):
            # linha paralela ao Z
            verts = np.array([[i,0,-T],[i,0,T]], dtype=float)
            ren.renderizar_wireframe(verts,[(0,1)],mat_id,view,proj,cor=cor_g)
            # linha paralela ao X
            verts = np.array([[-T,0,i],[T,0,i]], dtype=float)
            ren.renderizar_wireframe(verts,[(0,1)],mat_id,view,proj,cor=cor_g)

    def _draw_hud(self, fps: float):
        surf = self.surf
        forma = self.forma
        cam   = self.camera

        # ── Painel esquerdo (info do sólido) ─────────────────────────────
        rect_aa(surf, C_PANEL_DK, 10, 10, 280, 135, raio=12)
        rect_aa(surf, (0,0,0,0),  10, 10, 280, 135, raio=12, borda=1, cor_borda=C_BORDER)
        pygame.draw.rect(surf, forma.cor, (10, 10, 280, 4), border_radius=12)

        texto(surf, forma.nome, 22, 22, forma.cor, "Consolas", 18, bold=True)

        props = forma.propriedades
        py = 52
        for k in ("Vértices","Faces","Arestas","Tipo"):
            v = props.get(k,"—")
            texto(surf, f"{k}:", 22, py, C_TEXT_DIM, "Consolas", 12)
            texto(surf, str(v), 120, py, C_WHITE,    "Consolas", 12)
            py += 18

        # FPS
        texto(surf, f"FPS: {fps:5.1f}", 22, py+2, C_ACCENT, "Consolas", 13, bold=True)

        # ── Badge de modo (AUTO / MANUAL) ──────────────────────────────
        if self._modo_manual:
            badge_cor = C_ACCENT2
            badge_txt = "MODO: MANUAL"
        else:
            badge_cor = C_ACCENT
            badge_txt = "MODO: AUTO"
        rect_aa(surf, C_PANEL_DK, 10, 152, 280, 24, raio=8)
        rect_aa(surf, (0,0,0,0),  10, 152, 280, 24, raio=8, borda=1, cor_borda=badge_cor)
        texto(surf, badge_txt, 150, 157, badge_cor, "Consolas", 13, bold=True, centro=True)

        # ── Painel de Transformações (modo manual) ─────────────────────
        if self._modo_manual:
            self._draw_hud_manual()

        # ── Painel SLERP ───────────────────────────────────────────────
        slerp_t = self._manual_slerp_t if self._modo_manual else self._t_slerp
        alt_h = self.alt
        rect_aa(surf, C_PANEL_DK, 10, alt_h-64, 265, 52, raio=10)
        rect_aa(surf, (0,0,0,0),  10, alt_h-64, 265, 52, raio=10, borda=1, cor_borda=C_BORDER)
        lbl = "SLERP  (mouse drag)" if self._modo_manual else "SLERP  (interpolação de rotação)"
        texto(surf, lbl, 18, alt_h-58, C_TEXT_DIM, "Consolas", 11)

        bw = 230
        pygame.draw.rect(surf, (30,30,60), (18, alt_h-42, bw, 10), border_radius=5)
        fill_w = int(slerp_t * bw)
        if fill_w > 0:
            pygame.draw.rect(surf, C_ACCENT2, (18, alt_h-42, fill_w, 10), border_radius=5)
        pygame.draw.rect(surf, C_BORDER,    (18, alt_h-42, bw, 10), 1, border_radius=5)
        texto(surf, f"t={slerp_t:.2f}", 258, alt_h-44, C_ACCENT2, "Consolas", 11)

        # ── Controles (direita) ────────────────────────────────────────
        cx = self.larg - 200
        if self._modo_manual:
            ctrl = [
                ("MODO MANUAL",  C_ACCENT2,  True,  14),
                ("Mouse",        C_TEXT_DIM, False, 12),
                ("Scroll",       C_TEXT_DIM, False, 12),
                ("1-7",          C_TEXT_DIM, False, 12),
                ("Setas",        C_TEXT_DIM, False, 12),
                ("BACKSPACE",    C_TEXT_DIM, False, 12),
                ("TAB",          C_TEXT_DIM, False, 12),
                ("R",            C_TEXT_DIM, False, 12),
                ("ESC",          C_TEXT_DIM, False, 12),
            ]
            vals = [
                ("",                 C_ACCENT2,  True,  14),
                ("ajustar valor",    C_TEXT,     False, 12),
                ("eixo Z / fino",    C_TEXT,     False, 12),
                ("selecionar transf",C_TEXT,     False, 12),
                ("girar câmera",     C_TEXT,     False, 12),
                ("resetar transf",   C_TEXT,     False, 12),
                ("modo auto",        C_TEXT,     False, 12),
                ("resetar câmera",   C_TEXT,     False, 12),
                ("menu",             C_TEXT,     False, 12),
            ]
        else:
            ctrl = [
                ("CONTROLES",  C_ACCENT,   True,  14),
                ("W/S",        C_TEXT_DIM, False, 12),
                ("A/D",        C_TEXT_DIM, False, 12),
                ("Q/E",        C_TEXT_DIM, False, 12),
                ("Setas",      C_TEXT_DIM, False, 12),
                ("Z/X",        C_TEXT_DIM, False, 12),
                ("R",          C_TEXT_DIM, False, 12),
                ("TAB",        C_TEXT_DIM, False, 12),
                ("ESC",        C_TEXT_DIM, False, 12),
            ]
            vals = [
                ("",              C_ACCENT,   True,  14),
                ("frente / trás", C_TEXT,     False, 12),
                ("esq / dir",     C_TEXT,     False, 12),
                ("baixo / cima",  C_TEXT,     False, 12),
                ("girar câmera",  C_TEXT,     False, 12),
                ("zoom",          C_TEXT,     False, 12),
                ("resetar",       C_TEXT,     False, 12),
                ("modo manual",   C_TEXT,     False, 12),
                ("menu",          C_TEXT,     False, 12),
            ]
        rect_aa(surf, C_PANEL_DK, cx-10, 10, 200, 210, raio=10)
        rect_aa(surf, (0,0,0,0),  cx-10, 10, 200, 210, raio=10, borda=1, cor_borda=C_BORDER)

        cy2 = 20
        for (k, kc, kb, ks), (v, vc, vb, vs) in zip(ctrl, vals):
            texto(surf, k, cx+2,   cy2, kc, "Consolas", ks, bold=kb)
            texto(surf, v, cx+72,  cy2, vc, "Consolas", vs)
            cy2 += ks + 5

        # ── Câmera info ──────────────────────────────────────────────────
        cy3 = 230
        rect_aa(surf, C_PANEL_DK, cx-10, cy3, 200, 75, raio=10)
        rect_aa(surf, (0,0,0,0),  cx-10, cy3, 200, 75, raio=10, borda=1, cor_borda=C_BORDER)
        texto(surf, "CÂMERA", cx, cy3+8, C_ACCENT, "Consolas", 12, bold=True)
        px, py2, pz = cam.posicao
        texto(surf, f"Pos  ({px:+.1f},{py2:+.1f},{pz:+.1f})", cx, cy3+26, C_TEXT_DIM, "Consolas", 10)
        texto(surf, f"FOV  {cam.fov:.0f}°", cx, cy3+42, C_TEXT_DIM, "Consolas", 10)
        texto(surf, f"Yaw  {math.degrees(cam.yaw):.0f}°   Pitch {math.degrees(cam.pitch):.0f}°",
              cx, cy3+56, C_TEXT_DIM, "Consolas", 10)

        # ── Badge rodapé ──────────────────────────────────────────────────
        bw2 = 260 if self._modo_manual else 160
        rect_aa(surf, C_PANEL_MD, self.larg//2-bw2//2, self.alt-36, bw2, 26, raio=8)
        if self._modo_manual:
            texto(surf, "TAB -> auto  |  BACKSPACE -> reset  |  ESC -> menu",
                  self.larg//2, self.alt-29, C_TEXT_DIM, "Consolas", 10, centro=True)
        else:
            texto(surf, "TAB -> modo manual  |  ESC -> menu",
                  self.larg//2, self.alt-29, C_TEXT_DIM, "Consolas", 11, centro=True)

        if self._pausado and not self._modo_manual:
            texto(surf, "||  PAUSADO", self.larg//2, self.alt//2 - 20,
                  C_ACCENT, "Consolas", 22, bold=True, centro=True)

    # ── HUD do Modo Manual: cards de transformações clicáveis ──────────

    _TRANSF_NOMES = [
        (1, "Translação",    "TX TY TZ"),
        (2, "Escala",        "uniforme"),
        (3, "Rotação X",     "Euler"),
        (4, "Rotação Y",     "Euler"),
        (5, "Rotação Z",     "Euler"),
        (6, "SLERP",         "quaternion"),
        (7, "Cisalhamento",  "shear XY"),
    ]

    def _draw_hud_manual(self):
        """Desenha o painel lateral de transformações clicáveis."""
        surf = self.surf
        self._hud_cards = []

        card_x, card_w, card_h, gap = 10, 280, 34, 4
        start_y = 186

        for idx, (tid, nome, dica) in enumerate(self._TRANSF_NOMES):
            cy = start_y + idx * (card_h + gap)
            ativo = (tid == self._transf_ativa)
            rect = pygame.Rect(card_x, cy, card_w, card_h)
            self._hud_cards.append((rect, tid))

            # Fundo do card
            if ativo:
                cor_bg = lerp_color(C_PANEL_MD, C_ACCENT2, 0.18)
                cor_bd = C_ACCENT2
            else:
                cor_bg = C_PANEL_DK
                cor_bd = C_BORDER

            rect_aa(surf, cor_bg, card_x, cy, card_w, card_h, raio=8)
            rect_aa(surf, (0,0,0,0), card_x, cy, card_w, card_h,
                    raio=8, borda=1 if not ativo else 2, cor_borda=cor_bd)

            # Indicador lateral
            if ativo:
                pygame.draw.rect(surf, C_ACCENT2,
                                 (card_x, cy+6, 3, card_h-12), border_radius=2)

            # Número + Nome
            num_cor = C_ACCENT2 if ativo else C_TEXT_DIM
            nome_cor = C_WHITE if ativo else C_TEXT
            texto(surf, str(tid), card_x + 14, cy + 5, num_cor, "Consolas", 11, bold=True)
            texto(surf, nome, card_x + 32, cy + 5, nome_cor, "Consolas", 13, bold=ativo)

            # Valor atual
            val_str = self._valor_transf_str(tid)
            texto(surf, val_str, card_x + 32, cy + 20, C_TEXT_DIM, "Consolas", 10)

            # Dica à direita
            texto(surf, dica, card_x + card_w - 10, cy + 10,
                  C_TEXT_DIM, "Consolas", 9, centro=False)

    def _valor_transf_str(self, tid):
        """Retorna string formatada com o valor atual da transformação."""
        if tid == 1:
            return f"({self._manual_tx:+.2f}, {self._manual_ty:+.2f}, {self._manual_tz:+.2f})"
        elif tid == 2:
            return f"{self._manual_esc:.2f}x"
        elif tid == 3:
            return f"{math.degrees(self._manual_rx):+.1f}°"
        elif tid == 4:
            return f"{math.degrees(self._manual_ry):+.1f}°"
        elif tid == 5:
            return f"{math.degrees(self._manual_rz):+.1f}°"
        elif tid == 6:
            return f"t = {self._manual_slerp_t:.3f}"
        elif tid == 7:
            return f"a={self._manual_shear_a:+.2f}  b={self._manual_shear_b:+.2f}"
        return ""


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    pygame.init()
    pygame.display.set_caption("Visualizador 3D — Sólidos Geométricos | CG 2025")

    larg, alt = W, H
    surf = pygame.display.set_mode((larg, alt), pygame.RESIZABLE)
    clock = pygame.time.Clock()

    particulas = criar_particulas(60)
    menu = Menu(surf, particulas)
    visualizador = None
    estado = "menu"   # "menu" ou "viewer"

    rodando = True
    while rodando:
        dt = min(clock.tick(FPS) / 1000.0, 0.05)

        for p in particulas:
            p.update(dt, larg, alt)

        try:
            eventos = pygame.event.get()
        except Exception:
            # Contorna um bug conhecido no pygame padrão (SystemError em event.get)
            eventos = []

        for ev in eventos:
            if ev.type == pygame.QUIT:
                rodando = False

            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                if estado in ("viewer", "demo"):
                    estado = "menu"
                    visualizador = None
                elif estado == "menu":
                    rodando = False

            elif ev.type == pygame.VIDEORESIZE:
                larg, alt = ev.w, ev.h
                surf = pygame.display.set_mode((larg, alt), pygame.RESIZABLE)
                menu.surf = surf
                menu.larg = larg
                menu.alt  = alt
                if visualizador:
                    visualizador.surf = surf
                    visualizador.larg = larg
                    visualizador.alt  = alt
                    visualizador.renderer.largura = larg
                    visualizador.renderer.altura  = alt
                    visualizador.camera.aspect_ratio = larg / alt

            elif estado == "menu":
                resultado = menu.handle_event(ev)
                if resultado == "viewer":
                    forma = FORMAS[menu.selecionado - 1]  # offset -1 (demo eh item 0)
                    visualizador = Visualizador(surf, forma, larg, alt)
                    estado = "viewer"
                elif resultado == "demo":
                    visualizador = CenaDemo(surf, larg, alt)
                    estado = "demo"

            elif estado in ("viewer", "demo") and visualizador:
                resultado = visualizador.handle_event(ev)
                if resultado == "menu":
                    estado = "menu"
                    visualizador = None

        if estado == "menu":
            menu.update(dt)
            menu.draw(larg, alt)
        elif estado in ("viewer", "demo") and visualizador:
            visualizador.update(dt)
            visualizador.draw(fps=clock.get_fps())

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
