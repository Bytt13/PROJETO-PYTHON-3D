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

    Aceita tanto (superficie: pygame.Surface) quanto (width, height) como
    argumentos do construtor — conforme Listing 4 do enunciado.
    """

    def __init__(self, superficie_ou_largura=None, altura=None):
        """
        Inicializa o renderizador.

        Formas de uso:
            Renderizador(pygame_surface)       — usa a surface diretamente
            Renderizador(800, 600)             — cria imagem interna (conforme enunciado)
        """
        if isinstance(superficie_ou_largura, pygame.Surface):
            # Modo existente: recebe surface do pygame
            self.superficie = superficie_ou_largura
            self.largura = self.superficie.get_width()
            self.altura = self.superficie.get_height()
            self.image = None
        elif isinstance(superficie_ou_largura, (int, float)) and altura is not None:
            # Modo do enunciado: Renderizador(width, height)
            self.largura = int(superficie_ou_largura)
            self.altura = int(altura)
            # Cria imagem interna como np.zeros conforme enunciado
            self.image = np.zeros((self.altura, self.largura, 3))
            # Cria surface do pygame para rasterização
            self.superficie = pygame.Surface((self.largura, self.altura))
        else:
            # Fallback
            self.largura = 800
            self.altura = 600
            self.image = np.zeros((self.altura, self.largura, 3))
            self.superficie = pygame.Surface((self.largura, self.altura))

    # ════════════════════════════════════════════════════════════════════════
    # Utilitários internos
    # ════════════════════════════════════════════════════════════════════════

    def _clip_to_ndc_screen(self, v_clip: np.ndarray) -> Optional[Tuple[int,int]]:
        """
        Converte um vértice clip-space para pixel na tela.

        Passos 4, 5 e 6 do pipeline:
          4. Recorte (clipping) — descarta vértices fora do cubo NDC [-1, 1]
          5. Divisão por w       — converte coordenadas homogêneas para NDC
          6. Mapeamento viewport — converte NDC [-1,1] para pixels [0, largura/altura]

        Retorna None se o vértice está fora do frustum.
        """
        w = v_clip[3]
        # Passo 4: Clipping — vértice atrás da câmera
        if w <= 1e-6:
            return None
        # Passo 5: Divisão por w (perspectiva dividida)
        nx = v_clip[0] / w
        ny = v_clip[1] / w
        nz = v_clip[2] / w
        # Passo 4: Recorte no cubo normalizado (com margem para evitar pop-in)
        if nx < -2.0 or nx > 2.0 or ny < -2.0 or ny > 2.0 or nz < -1.0 or nz > 1.0:
            return None
        # Passo 6: Mapeamento para coordenadas do dispositivo (viewport)
        px = int((nx + 1.0) * 0.5 * self.largura)
        py = int((1.0 - ny) * 0.5 * self.altura)
        return (px, py)

    def _transform_vertices(self, vertices, matrix):
        """
        Aplica transformação a vértices (conforme Listing 4 do enunciado).

        Converte vértices (N, 3) para coordenadas homogêneas (N, 4),
        multiplica pela matriz de transformação e retorna (N, 4).

        Args:
            vertices: array (N, 3) de pontos 3D
            matrix: matriz de transformação 4×4

        Returns:
            array (N, 4) de vértices transformados em coordenadas homogêneas
        """
        n = len(vertices)
        verts_h = np.ones((n, 4), dtype=float)
        verts_h[:, :3] = vertices[:, :3] if vertices.shape[1] >= 3 else vertices
        return (matrix @ verts_h.T).T

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
        # Fill light (luz de preenchimento — suaviza sombras)
        fill_light_dir = np.array([-0.4, -0.6, 0.5], dtype=float)
        fill_light_dir = fill_light_dir / (np.linalg.norm(fill_light_dir) + 1e-10)

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

            # ── Backface Culling ───────────────────────────────────────────
            # A normal da face vem do winding dos vértices (CCW = outward).
            # Se a normal aponta para o mesmo lado que o vetor câmera→face,
            # estamos vendo o VERSO da face → descarta.
            centroid_w = np.mean(world_verts[[i for i in face], :3], axis=0)
            view_dir_to_face = centroid_w - camera_pos
            if np.dot(normal, view_dir_to_face) >= 0:
                continue

            # ── Profundidade (Z em view space para ordenação) ──────────────
            # Usa a profundidade MÁXIMA (mais distante) da face para
            # reduzir artefatos de Z-fighting no algoritmo do pintor.
            depth = float(np.min([view_verts[i, 2] for i in face]))

            # ── Iluminação Blinn-Phong aprimorada ─────────────────────────
            # Vetor da câmera para a face
            v_dir = camera_pos - centroid_w
            v_len = np.linalg.norm(v_dir)
            if v_len > 1e-10:
                v_dir = v_dir / v_len

            # --- Luz principal (key light) ---
            diff = max(0.0, np.dot(normal, luz))

            # Halfway vector para especular
            H = luz + v_dir
            H_len = np.linalg.norm(H)
            H = H / H_len if H_len > 1e-10 else H
            spec = max(0.0, np.dot(normal, H)) ** 48

            # --- Fill light (luz de preenchimento inferior-esquerda) ---
            # Ilumina as sombras suavemente para não ficarem completamente escuras
            diff_fill = max(0.0, np.dot(normal, fill_light_dir)) * 0.25

            # --- Rim light (efeito Fresnel) ---
            # Faces quase perpendiculares ao olhar ganham um brilho sutil na borda
            ndotv = max(0.0, np.dot(normal, v_dir))
            rim = (1.0 - ndotv) ** 3 * 0.30

            # Combinação final de iluminação
            Ka = 0.15    # ambiente base
            Kd = 0.62    # difuso (key light)
            Ks = 0.35    # especular
            intensity = Ka + Kd * diff + Ks * spec + diff_fill + rim
            intensity = min(1.0, max(0.0, intensity))

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
                # Borda anti-aliased com cor misturada (mais sutil e elegante)
                edge = (
                    max(0, int(fill[0] * 0.55)),
                    max(0, int(fill[1] * 0.55)),
                    max(0, int(fill[2] * 0.55)),
                )
                pygame.draw.aalines(self.superficie, edge, True, pts)

    # ════════════════════════════════════════════════════════════════════════
    # MODO WIREFRAME — Rasterização com Bresenham
    # ════════════════════════════════════════════════════════════════════════

    def _draw_line(self, p1, p2, cor):
        """
        Desenha linha entre dois pontos na tela (algoritmo de Bresenham).

        Alias conforme Listing 4 do enunciado.

        Args:
            p1: tupla (x, y) do ponto inicial
            p2: tupla (x, y) do ponto final
            cor: tupla RGB da cor
        """
        self.bresenham_linha(p1[0], p1[1], p2[0], p2[1], cor)

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

    # ════════════════════════════════════════════════════════════════════════
    # MÉTODO render() — Interface conforme enunciado (Listing 4)
    # ════════════════════════════════════════════════════════════════════════

    def render(self, objetos, camera):
        """
        Pipeline completo de renderização — conforme Listing 4 do enunciado.

        Executa os 7 passos do pipeline para cada objeto:
          1. Obter matrizes da câmera (View e Projection)
          2. Transformação Model-View-Projection
          3. Transformar vértices para coordenadas normalizadas
          4. Recorte (clipping) no cubo normalizado [-1,1]
          5. Perspectiva dividida (divisão por w)
          6. Mapeamento para coordenadas do dispositivo (viewport)
          7. Rasterização das arestas (Bresenham 3D)

        Args:
            objetos: lista de Objeto3D
            camera: instância de Camera
        """
        # Passo 1: Obter matrizes da câmera
        V = camera.get_view_matrix()
        P = camera.get_projection_matrix()

        for obj in objetos:
            # Passo 2: Transformação Model-View-Projection
            mvp = P @ V @ obj.model_matrix

            # Passo 3: Transformar vértices para coordenadas clip
            vertices_clip = self._transform_vertices(obj.vertices, mvp)

            # Converter cor de [0,1] para [0,255] se necessário
            cor = obj.cor
            if isinstance(cor, (list, tuple)) and len(cor) >= 3:
                if all(isinstance(c, float) and c <= 1.0 for c in cor):
                    cor = (int(cor[0]*255), int(cor[1]*255), int(cor[2]*255))

            # Passos 4-7: Clipping, divisão por w, viewport, rasterização
            for i, j in obj.arestas:
                v1, v2 = vertices_clip[i], vertices_clip[j]
                p1 = self._clip_to_ndc_screen(v1)
                p2 = self._clip_to_ndc_screen(v2)
                if p1 and p2:
                    self.bresenham_linha(p1[0], p1[1], p2[0], p2[1], cor)
