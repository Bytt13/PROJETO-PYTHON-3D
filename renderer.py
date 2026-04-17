"""
=============================================================================
 PARTE 4: PIPELINE DE RENDERIZAÇÃO (O MOTOR GRÁFICO)
=============================================================================

 Os 7 passos clássicos do pipeline de renderização software:
   1. Transformação de Modelo  (Local → Mundo)
   2. Transformação de Visão   (Mundo → Câmera)
   3. Projeção                 (Câmera → Clip)
   4. Recorte (Clipping)
   5. Divisão por W            (Clip → NDC)
   6. Mapeamento de Viewport   (NDC → Pixels)
   7. Rasterização             (Bresenham para wireframe / Painter para sólido)

 RENDERIZAÇÃO SÓLIDA — ALGORITMO DO PINTOR + BLINN-PHONG:
   1. Calcula a normal de cada face em espaço de mundo.
   2. Aplica backface culling (descarta faces de costas para câmera).
   3. Ordena faces de mais distante para mais próxima (algoritmo do pintor).
   4. Pinta cada face com iluminação Blinn-Phong (ambient + diffuse + specular).
   5. Desenha bordas escuras sobre as faces para definição.

=============================================================================
"""

import numpy as np
import pygame
from typing import List, Tuple, Optional


class Renderizador:
    """
    Motor de renderização 3D software (sem OpenGL).
    Suporta modo wireframe (Bresenham) e modo sólido (Painter + Phong).
    """

    def __init__(self, superficie: pygame.Surface):
        self.superficie = superficie
        self.largura = superficie.get_width()
        self.altura = superficie.get_height()

    # ════════════════════════════════════════════════════════════════════════
    # Utilitários internos
    # ════════════════════════════════════════════════════════════════════════

    def _clip_to_ndc_screen(self, v_clip: np.ndarray) -> Optional[Tuple[int,int]]:
        """Converte um vértice clip-space para pixel. None se fora da câmera."""
        w = v_clip[3]
        if w <= 1e-6:
            return None
        nx = v_clip[0] / w
        ny = v_clip[1] / w
        px = int((nx + 1.0) * 0.5 * self.largura)
        py = int((1.0 - ny) * 0.5 * self.altura)
        return (px, py)

    @staticmethod
    def _face_normal(world_verts: np.ndarray, face: Tuple) -> np.ndarray:
        """
        Computa a normal de uma face poligonal em espaço de mundo.
        Usa os três primeiros vértices (Newell method simplificado).
        """
        v0 = world_verts[face[0], :3]
        v1 = world_verts[face[1], :3]
        v2 = world_verts[face[2], :3]
        n = np.cross(v1 - v0, v2 - v0)
        length = np.linalg.norm(n)
        return n / length if length > 1e-10 else np.zeros(3)

    # ════════════════════════════════════════════════════════════════════════
    # MODO SÓLIDO — Algoritmo do Pintor + Iluminação Blinn-Phong
    # ════════════════════════════════════════════════════════════════════════

    def renderizar_solido(
        self,
        vertices: np.ndarray,
        faces: List[Tuple],
        mat_modelo: np.ndarray,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
        cor_base: Tuple[int, int, int],
        camera_pos: np.ndarray,
        luz_dir: np.ndarray = None,
        bordas: bool = True,
    ):
        """
        Renderiza um sólido 3D com faces preenchidas e iluminação.

        ILUMINAÇÃO BLINN-PHONG (flat shading por face):
          I = Ka + Kd·max(0, N·L) + Ks·max(0, N·H)^shininess

          Ka = componente ambiente   (luz mínima mesmo na sombra)
          Kd = componente difusa     (Lambertiana — função do ângulo N·L)
          Ks = componente especular  (brilho de superfície — N·H elevado a shininess)
          N  = normal da face (espaço de mundo)
          L  = direção da luz (normalizada)
          H  = halfway vector = normalize(L + V)  [Blinn-Phong trick]
          V  = direção da câmera ao centroide da face

        ALGORITMO DO PINTOR:
          Ordena faces do mais distante para o mais próximo em Z de view.
          Desenha nessa ordem → faces próximas cobrem as distantes.
          Simples e eficaz para objetos convexos.
        """
        if luz_dir is None:
            luz_dir = np.array([0.6, 1.0, -0.8], dtype=float)
        luz = luz_dir / (np.linalg.norm(luz_dir) + 1e-10)

        n = len(vertices)
        verts_h = np.ones((n, 4), dtype=float)
        verts_h[:, :3] = vertices

        # Espaços de transformação
        model_view = view_matrix @ mat_modelo
        mvp = proj_matrix @ model_view

        world_verts  = (mat_modelo   @ verts_h.T).T   # (N,4) — normais e backface
        view_verts   = (model_view   @ verts_h.T).T   # (N,4) — depth sorting
        clip_verts   = (mvp          @ verts_h.T).T   # (N,4) — tela

        model_center = mat_modelo[:3, 3]   # origem do objeto em espaço-mundo

        drawable = []

        for face in faces:
            # ── Coordenadas de tela ────────────────────────────────────────
            screen_pts = []
            any_behind = False
            for idx in face:
                pt = self._clip_to_ndc_screen(clip_verts[idx])
                if pt is None:
                    any_behind = True
                    break
                screen_pts.append(pt)
            if any_behind or len(screen_pts) < 3:
                continue

            # ── Normal em espaço-mundo ─────────────────────────────────────
            normal = self._face_normal(world_verts, face)
            if np.linalg.norm(normal) < 1e-10:
                continue

            # Garante que a normal aponta para FORA do sólido
            centroid_w = np.mean(world_verts[[i for i in face], :3], axis=0)
            if np.dot(normal, centroid_w - model_center) < 0:
                normal = -normal

            # ── Backface Culling ───────────────────────────────────────────
            # Se a normal aponta para o mesmo lado que o vetor câmera→face,
            # estamos vendo o VERSO da face → descarta.
            view_dir_to_face = centroid_w - camera_pos
            if np.dot(normal, view_dir_to_face) >= 0:
                continue

            # ── Profundidade (Z em view space para ordenação) ──────────────
            depth = float(np.mean([view_verts[i, 2] for i in face]))

            # ── Iluminação Blinn-Phong ─────────────────────────────────────
            # Componente difusa (Lambertiana)
            diff = max(0.0, np.dot(normal, luz))

            # Componente especular (Blinn-Phong halfway vector)
            v_dir = camera_pos - centroid_w
            v_len = np.linalg.norm(v_dir)
            if v_len > 1e-10:
                v_dir = v_dir / v_len
            H = luz + v_dir
            H_len = np.linalg.norm(H)
            H = H / H_len if H_len > 1e-10 else H
            spec = max(0.0, np.dot(normal, H)) ** 64    # shininess = 64

            # Combinação final
            Ka = 0.18   # ambiente
            Kd = 0.70   # difuso
            Ks = 0.45   # especular
            intensity = Ka + Kd * diff + Ks * spec
            intensity = min(1.0, intensity)

            fill = (
                min(255, int(cor_base[0] * intensity)),
                min(255, int(cor_base[1] * intensity)),
                min(255, int(cor_base[2] * intensity)),
            )
            drawable.append((depth, screen_pts, fill, normal))

        # ── Algoritmo do Pintor: mais distante primeiro ────────────────────
        # Em view space, z cresce negativamente na frente da câmera.
        # Mais distante = Z menor (mais negativo) → order ASCENDING z.
        drawable.sort(key=lambda x: x[0])

        # ── Rasterização ───────────────────────────────────────────────────
        for depth, pts, fill, normal in drawable:
            pygame.draw.polygon(self.superficie, fill, pts)
            if bordas:
                # Borda ligeiramente mais escura para definição de arestas
                edge = (
                    max(0, fill[0] - 55),
                    max(0, fill[1] - 55),
                    max(0, fill[2] - 55),
                )
                pygame.draw.polygon(self.superficie, edge, pts, 1)

    # ════════════════════════════════════════════════════════════════════════
    # MODO WIREFRAME — Rasterização com Bresenham
    # ════════════════════════════════════════════════════════════════════════

    def bresenham_linha(self, x0:int, y0:int, x1:int, y1:int, cor:Tuple):
        """
        Algoritmo de Bresenham (1962): traça uma linha pixel a pixel
        usando apenas adições de inteiros — sem divisão ou ponto flutuante.

        Mantém um 'erro acumulado'. Quando ultrapassa o limiar,
        avança no eixo secundário. Garante a linha mais próxima
        da reta ideal entre dois pixels.
        """
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        erro = dx - dy
        w, h = self.largura, self.altura
        while True:
            if 0 <= x0 < w and 0 <= y0 < h:
                self.superficie.set_at((x0, y0), cor)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * erro
            if e2 > -dy:
                erro -= dy
                x0 += sx
            if e2 < dx:
                erro += dx
                y0 += sy

    def renderizar_wireframe(
        self,
        vertices: np.ndarray,
        arestas: List[Tuple[int,int]],
        mat_modelo: np.ndarray,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
        cor: Tuple = (255, 255, 255),
    ):
        """Renderiza um objeto em wireframe via Bresenham."""
        n = len(vertices)
        verts_h = np.ones((n, 4), dtype=float)
        verts_h[:, :3] = vertices
        mvp = proj_matrix @ view_matrix @ mat_modelo
        clip = (mvp @ verts_h.T).T

        for i, j in arestas:
            v1, v2 = clip[i], clip[j]
            p1 = self._clip_to_ndc_screen(v1)
            p2 = self._clip_to_ndc_screen(v2)
            if p1 and p2:
                self.bresenham_linha(p1[0], p1[1], p2[0], p2[1], cor)
