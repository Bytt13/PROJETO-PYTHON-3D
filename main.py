"""
=============================================================================
 PARTE 5: APLICAÇÃO PRINCIPAL — MENU + VISUALIZADOR 3D SÓLIDO
=============================================================================

 Controles no MENU:
   ↑ / ↓      → navegar entre os sólidos
   ENTER      → entrar no visualizador

 Controles no VISUALIZADOR:
   W/S        → mover câmera frente/trás
   A/D        → mover câmera esquerda/direita
   Q/E        → mover câmera baixo/cima
   ↑/↓/←/→   → girar câmera
   Z/X        → zoom (ajustar FOV)
   R          → resetar câmera
   ESPAÇO     → pausar rotação automática
   ESC        → voltar ao menu

=============================================================================
"""

import sys, math, random
import numpy as np
import pygame

from transforms import translacao, escala, rotacao_x, rotacao_y, rotacao_z, compor
from quaternion import Quaternion, slerp as q_slerp
from camera import Camera
from renderer import Renderizador
from mesh import FORMAS, Mesh

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
        self.n = len(FORMAS)
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
                return "viewer"
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            mx, my = ev.pos
            for i in range(self.n):
                cy = self._card_y(i)
                if self.LIST_X <= mx <= self.LIST_X + self.LIST_W and cy <= my <= cy + self.CARD_H:
                    if self.selecionado == i:
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

        # ── Lista de sólidos (esquerda) ────────────────────────────────────
        for i, forma in enumerate(FORMAS):
            cy = self._card_y(i)
            selecionado = (i == self.selecionado)

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
                texto(surf, "▶", self.LIST_X + self.LIST_W - 28, cy + 21,
                      cor_titulo, "Consolas", 16)

        # ── Painel de descrição (direita) ──────────────────────────────────────────
        forma = FORMAS[self.selecionado]
        dx, dy = self.DESC_X, 120
        dw = larg - self.DESC_X - 50
        dh = alt - 180

        rect_aa(surf, C_PANEL_DK, dx, dy-10, dw, dh, raio=14)
        rect_aa(surf, (0,0,0,0), dx, dy-10, dw, dh, raio=14, borda=1, cor_borda=C_BORDER)

        # Patch colorido no topo do painel
        pygame.draw.rect(surf, forma.cor,
                         (dx, dy-10, dw, 5), border_radius=14)

        # Nome grande
        texto(surf, forma.nome, dx + dw//2, dy + 10,
              forma.cor, "Consolas", 30, bold=True, centro=True)

        # Badge tipo
        tipo = forma.propriedades.get("Tipo", "")
        tw, _ = texto(surf, tipo, dx + dw//2, dy + 52,
                      C_TEXT_DIM, "Consolas", 13, centro=True)
        rect_aa(surf, C_PANEL_MD, dx + dw//2 - tw//2 - 10, dy + 48,
                tw + 20, 20, raio=10)
        texto(surf, tipo, dx + dw//2, dy + 52,
              C_TEXT_DIM, "Consolas", 13, centro=True)

        # Linha
        pygame.draw.line(surf, C_BORDER,
                         (dx+20, dy+78), (dx+dw-20, dy+78))

        # Propriedades em tabela
        ry = dy + 92
        for k, v in forma.propriedades.items():
            if k == "Tipo":
                continue
            texto(surf, k, dx + 30, ry, C_TEXT_DIM, "Consolas", 13)
            texto(surf, str(v), dx + dw - 30, ry, C_WHITE, "Consolas", 13,
                  centro=False)
            # Alinha valor à direita
            fw, _ = texto(surf, str(v), dx, ry, C_WHITE, "Consolas", 13)
            surf.fill((0,0,0,0), (dx, ry, fw, 18))  # apaga
            texto(surf, str(v), dx + dw - 30 - fw, ry, C_WHITE, "Consolas", 13)
            ry += 22

        pygame.draw.line(surf, C_BORDER,
                         (dx+20, ry+4), (dx+dw-20, ry+4))

        # Descrição (multi-linha)
        ry += 16
        for linha in forma.descricao.split("\n"):
            texto(surf, linha, dx + 30, ry, C_TEXT, "Consolas", 13)
            ry += 20

        # Botão ENTER
        btn_y = dy + dh - 52
        pulse2 = (math.sin(self._pulse * 1.5) + 1) / 2
        btn_cor = lerp_color(C_PANEL_MD, forma.cor, pulse2 * 0.3)
        rect_aa(surf, btn_cor, dx + 40, btn_y, dw - 80, 36, raio=10)
        rect_aa(surf, (0,0,0), dx + 40, btn_y, dw - 80, 36,
                raio=10, borda=2, cor_borda=lerp_color(C_BORDER, forma.cor, pulse2))
        texto(surf, "▷  ENTER   para visualizar",
              dx + dw//2, btn_y + 9, C_WHITE, "Consolas", 15, bold=True, centro=True)

        # ── Rodápé ─────────────────────────────────────────────────────────────────
        pygame.draw.line(surf, C_BORDER, (0, alt-36), (larg, alt-36))
        texto(surf, "↑↓  Navegar    ENTER  Visualizar    ESC  Sair",
              larg//2, alt-26, C_TEXT_DIM, "Consolas", 12, centro=True)


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

        dist = max(3.5, float(np.max(np.abs(forma.vertices))) * 3.2)
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

    def handle_event(self, ev):
        """Retorna 'menu' se ESC, None caso contrário."""
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                return "menu"
            if ev.key == pygame.K_SPACE:
                self._pausado = not self._pausado
            if ev.key == pygame.K_r:
                dist = max(3.5, float(np.max(np.abs(self.forma.vertices))) * 3.2)
                self.camera.posicao = np.array([0.0, 1.8, dist])
                self.camera.yaw   = -math.pi / 2
                self.camera.pitch = -0.22
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

        # ── Matriz de modelo: aplica rotação do quatérnio ──────────────────
        # Combina rotação automática com toque do SLERP
        q_slerp_atual = q_slerp(self._q_A, self._q_B, self._t_slerp)
        q_total = self._q_rot * q_slerp_atual
        mat_rot = q_total.to_matrix()

        # Escala para que objetos muito grandes ou pequenos caibam bem
        escala_auto = 1.3 / max(0.01, float(np.max(np.abs(forma.vertices))))
        mat_modelo = compor(mat_rot, escala(escala_auto, escala_auto, escala_auto))

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

        # ── Painel esquerdo ─────────────────────────────────────────────────
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

        # ── Painel SLERP ───────────────────────────────────────────────────
        rect_aa(surf, C_PANEL_DK, 10, H-64, 265, 52, raio=10)
        rect_aa(surf, (0,0,0,0),  10, H-64, 265, 52, raio=10, borda=1, cor_borda=C_BORDER)
        texto(surf, "SLERP  (interpolação de rotação)",
              18, H-58, C_TEXT_DIM, "Consolas", 11)

        bw = 230
        pygame.draw.rect(surf, (30,30,60), (18, H-42, bw, 10), border_radius=5)
        fill_w = int(self._t_slerp * bw)
        if fill_w > 0:
            pygame.draw.rect(surf, C_ACCENT2, (18, H-42, fill_w, 10), border_radius=5)
        pygame.draw.rect(surf, C_BORDER,    (18, H-42, bw, 10), 1, border_radius=5)
        texto(surf, f"t={self._t_slerp:.2f}", 258, H-44, C_ACCENT2, "Consolas", 11)

        # ── Controles (direita) ─────────────────────────────────────────────
        cx = self.larg - 200
        ctrl = [
            ("CONTROLES",  C_ACCENT,   True,  14),
            ("W/S",        C_TEXT_DIM, False, 12),
            ("A/D",        C_TEXT_DIM, False, 12),
            ("Q/E",        C_TEXT_DIM, False, 12),
            ("↑↓←→",      C_TEXT_DIM, False, 12),
            ("Z/X",        C_TEXT_DIM, False, 12),
            ("R",          C_TEXT_DIM, False, 12),
            ("ESPAÇO",     C_TEXT_DIM, False, 12),
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
            ("pausar",        C_TEXT,     False, 12),
            ("menu",          C_TEXT,     False, 12),
        ]
        rect_aa(surf, C_PANEL_DK, cx-10, 10, 200, 210, raio=10)
        rect_aa(surf, (0,0,0,0),  cx-10, 10, 200, 210, raio=10, borda=1, cor_borda=C_BORDER)

        cy2 = 20
        for (k, kc, kb, ks), (v, vc, vb, vs) in zip(ctrl, vals):
            texto(surf, k, cx+2,   cy2, kc, "Consolas", ks, bold=kb)
            texto(surf, v, cx+72,  cy2, vc, "Consolas", vs)
            cy2 += ks + 5

        # ── Câmera info ─────────────────────────────────────────────────────
        cy3 = 230
        rect_aa(surf, C_PANEL_DK, cx-10, cy3, 200, 75, raio=10)
        rect_aa(surf, (0,0,0,0),  cx-10, cy3, 200, 75, raio=10, borda=1, cor_borda=C_BORDER)
        texto(surf, "CÂMERA", cx, cy3+8, C_ACCENT, "Consolas", 12, bold=True)
        px, py2, pz = cam.posicao
        texto(surf, f"Pos  ({px:+.1f},{py2:+.1f},{pz:+.1f})", cx, cy3+26, C_TEXT_DIM, "Consolas", 10)
        texto(surf, f"FOV  {cam.fov:.0f}°", cx, cy3+42, C_TEXT_DIM, "Consolas", 10)
        texto(surf, f"Yaw  {math.degrees(cam.yaw):.0f}°   Pitch {math.degrees(cam.pitch):.0f}°",
              cx, cy3+56, C_TEXT_DIM, "Consolas", 10)

        # ── Badge ESC ──────────────────────────────────────────────────────
        rect_aa(surf, C_PANEL_MD, self.larg//2-80, self.alt-36, 160, 26, raio=8)
        texto(surf, "ESC → voltar ao menu",
              self.larg//2, self.alt-29, C_TEXT_DIM, "Consolas", 12, centro=True)

        if self._pausado:
            texto(surf, "⏸  PAUSADO", self.larg//2, self.alt//2 - 20,
                  C_ACCENT, "Consolas", 22, bold=True, centro=True)


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

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                rodando = False

            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                if estado == "viewer":
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
                    forma = FORMAS[menu.selecionado]
                    visualizador = Visualizador(surf, forma, larg, alt)
                    estado = "viewer"

            elif estado == "viewer" and visualizador:
                resultado = visualizador.handle_event(ev)
                if resultado == "menu":
                    estado = "menu"
                    visualizador = None

        if estado == "menu":
            menu.update(dt)
            menu.draw(larg, alt)
        elif estado == "viewer" and visualizador:
            visualizador.update(dt)
            visualizador.draw(fps=clock.get_fps())

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
