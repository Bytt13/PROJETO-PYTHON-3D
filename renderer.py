"""
=============================================================================
 PARTE 4: PIPELINE DE RENDERIZAÇÃO (O MOTOR GRÁFICO)
=============================================================================

 ╔══════════════════════════════════════════════════════════════════╗
 ║  OS 7 PASSOS DO PIPELINE (A LINHA DE MONTAGEM 3D→2D)            ║
 ╚══════════════════════════════════════════════════════════════════╝

 ANALOGIA DA FÁBRICA:
 Imagine uma linha de montagem de carros. Cada estação faz UMA coisa
 específica antes de passar para a próxima. O resultado final é um carro
 pronto. Aqui, o "produto" são pixels na tela.

 PASSO 1 — Transformação de Modelo (Local → Mundo):
   Cada objeto 3D existe em seu próprio espaço local.
   Um cubo centrado em [0,0,0] é multiplicado pela Matriz de Modelo
   para posicioná-lo, girar e escalar ele no mundo.

 PASSO 2 — Transformação de Visão (Mundo → Câmera):
   A Matriz de Visão move o mundo inteiro para que a câmera
   fique na origem olhando para -Z. Simplifica os próximos passos.

 PASSO 3 — Projeção (Câmera → Cubo Canônico):
   A Matriz de Projeção deforma o frustum (cone de visão) em um
   cubo perfeito de coordenadas -1 a +1 chamado NDC.
   Objetos distantes são comprimidos; próximos, expandidos.

 PASSO 4 — Recorte (Clipping):
   Descarta tudo que está fora do frustum (fora da tela).
   ANALOGIA: é a tesoura que corta o que sai da foto.

 PASSO 5 — Divisão por W (Divisão de Perspectiva):
   Divide x, y, z pelo componente w de cada vértice.
   É aqui que a mágica da profundidade acontece:
   objetos longe têm w grande → divididos → aparecem menores!

 PASSO 6 — Mapeamento de Viewport (NDC → Pixels):
   Converte coordenadas -1/+1 para pixels reais da janela.
   [-1,+1] → [0, largura] e [-1,+1] → [0, altura]

 PASSO 7 — Rasterização (Pixels!):
   O algoritmo de Bresenham decide quais pixels colorir
   para desenhar linhas e triângulos.

 ╔══════════════════════════════════════════════════════════════════╗
 ║  O ALGORITMO DE BRESENHAM (QUAL PIXEL PINTAR?)                  ║
 ╚══════════════════════════════════════════════════════════════════╝

 PROBLEMA: Uma linha entre (0,0) e (7,3) passa por infinitos pontos reais.
 Mas pixels são discretos! Temos que escolher os pixels "mais próximos".

 ANALOGIA DO FOTÓGRAFO:
 Imagine uma grade de xadrez. Você quer riscar uma linha diagonal com giz.
 O Bresenham é como um fotógrafo que decide: "este quadrado está MAIS perto
 da linha real? Então pinto ele."

 O TRUQUE: em vez de calcular raiz quadrada ou divisão (lento!),
 o Bresenham mantém um "erro acumulado" inteiro. Quando o erro ultrapassa
 o threshold, ele move para o próximo pixel na perpendicular.
 
 Era revolucionário em 1962 — processadores lentos não podiam fazer
 operações de ponto flutuante. Bresenham usava só inteiros e deslocamentos!

=============================================================================
"""

import numpy as np
import pygame
from typing import List, Tuple, Optional


class Renderizador:
    """
    Motor de renderização 3D software (sem OpenGL).

    Executa o pipeline completo: Modelo → Visão → Projeção →
    Clipping → Divisão-W → Viewport → Rasterização.
    """

    def __init__(self, superficie: pygame.Surface):
        """
        Inicializa o renderizador.

        Args:
            superficie: Surface do Pygame onde renderizar
        """
        self.superficie = superficie
        self.largura = superficie.get_width()
        self.altura = superficie.get_height()

    # ════════════════════════════════════════════════════════════════════════
    # PASSO 1-3: Transformações combinadas (Modelo, Visão, Projeção)
    # ════════════════════════════════════════════════════════════════════════

    def transformar_vertices(
        self,
        vertices: np.ndarray,
        matriz_modelo: np.ndarray,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica o pipeline de transformação a todos os vértices.

        O pipeline MVP (Model-View-Projection) é a espinha dorsal
        de qualquer motor gráfico. Cada vértice passa pelas 3 matrizes:

          v_clip = Proj × View × Model × v_local

        Fazemos isso de uma vez ao multiplicar as matrizes primeiro
        (MVP = Proj × View × Model) e depois aplicando a todos os vértices.
        Muito mais eficiente se há muitos vértices!

        Args:
            vertices: array Nx3 de vértices locais do objeto
            matriz_modelo: posição/rotação/escala no mundo
            view_matrix: transforma para espaço da câmera
            proj_matrix: aplica perspectiva

        Returns:
            (vertices_clip, vertices_ndc) — ambos em coordenadas homogêneas Nx4
        """
        # Converte para coordenadas homogêneas: adiciona w=1
        n = len(vertices)
        verts_h = np.ones((n, 4), dtype=float)
        verts_h[:, :3] = vertices

        # Matriz MVP combinada (Da direita para esquerda: Model → View → Proj)
        mvp = proj_matrix @ view_matrix @ matriz_modelo

        # Aplica MVP a todos os vértices de uma vez (transpõe para broadcast)
        # Resultado: cada coluna = um vértice transformado
        clip_coords = (mvp @ verts_h.T).T  # shape: (N, 4)

        return clip_coords

    # ════════════════════════════════════════════════════════════════════════
    # PASSO 4: Clipping (Recorte no espaço do Frustum)
    # ════════════════════════════════════════════════════════════════════════

    def _dentro_frustum(self, v: np.ndarray) -> bool:
        """
        Verifica se um vértice está dentro do frustum (cubo NDC -1 a +1).

        Para ser visível, todas as coordenadas clip devem satisfazer:
          -w ≤ x ≤ w
          -w ≤ y ≤ w
          -w ≤ z ≤ w  (com w > 0)

        ANALOGIA: É como checar se um ponto está dentro de uma caixa.
        Fora = descarte (recorte); dentro = processa.
        """
        w = v[3]
        if w <= 0:
            return False  # Atrás da câmera
        return (
            -w <= v[0] <= w and
            -w <= v[1] <= w and
            -w <= v[2] <= w
        )

    def recortar_aresta(
        self,
        v1_clip: np.ndarray,
        v2_clip: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Recorte de uma aresta no frustum (Cohen-Sutherland simplificado).

        Se ambos os vértices estão fora do mesmo lado do frustum,
        a aresta toda é descartada. Caso contrário, pelo menos
        parte pode ser visível.

        Para este motor simplificado: se qualquer vértice está fora,
        verificamos se a aresta cruza o plano near (z=-w, atrás da câmera).

        Returns:
            (v1, v2) clipped ou None se a aresta é completamente invisível
        """
        dentro1 = self._dentro_frustum(v1_clip)
        dentro2 = self._dentro_frustum(v2_clip)

        if dentro1 and dentro2:
            return v1_clip, v2_clip

        if not dentro1 and not dentro2:
            return None  # Totalmente fora

        # Um dentro, um fora: interpola para encontrar o ponto de cruzamento
        # com o plano near (w + z = 0, ou seja, z = -w)
        if not dentro1:
            v1_clip, v2_clip = v2_clip, v1_clip  # Garante que v1 está dentro

        # Fator de interpolação para o plano near
        w1, w2 = v1_clip[3], v2_clip[3]
        z1, z2 = v1_clip[2], v2_clip[2]
        denom = (w2 - w1) - (z2 - z1)
        if abs(denom) < 1e-10:
            return None

        t = (w1 - z1) / denom
        t = np.clip(t, 0, 1)
        v_medio = v1_clip + t * (v2_clip - v1_clip)

        return v1_clip, v_medio

    # ════════════════════════════════════════════════════════════════════════
    # PASSO 5: Divisão por W (Perspectiva Dividida)
    # ════════════════════════════════════════════════════════════════════════

    def divisao_perspectiva(self, v_clip: np.ndarray) -> Optional[np.ndarray]:
        """
        Divide x, y, z pelo componente w para obter coordenadas NDC.

        NDC = Normalized Device Coordinates: valores de -1 a +1.

        MAGIA DA PERSPECTIVA:
        O componente w da projeção perspectiva é -z (distância da câmera).
        Ao dividir por w, x e y são automaticamente diminuídos para
        objetos distantes e aumentados para objetos próximos.
        
        É matematicamente equivalente à projeção perspectiva clássica:
          x_tela = x_olho / (-z_olho) × distância_focal
          
        Args:
            v_clip: vértice em coordenadas clip [x, y, z, w]

        Returns:
            coordenadas NDC [x_ndc, y_ndc, z_ndc] ou None se w≤0
        """
        w = v_clip[3]
        if abs(w) < 1e-10:
            return None

        return np.array([
            v_clip[0] / w,  # x_ndc: -1 (esquerda) a +1 (direita)
            v_clip[1] / w,  # y_ndc: -1 (baixo) a +1 (cima)
            v_clip[2] / w,  # z_ndc: -1 (perto) a +1 (longe) — depth
        ])

    # ════════════════════════════════════════════════════════════════════════
    # PASSO 6: Mapeamento de Viewport
    # ════════════════════════════════════════════════════════════════════════

    def ndc_para_viewport(self, ndc: np.ndarray) -> Tuple[int, int]:
        """
        Converte coordenadas NDC [-1, +1] para pixels da tela.

        ANALOGIA: é como esticar uma foto para caber em um moldura específica.

        NDC x=-1 → pixel 0 (extrema esquerda)
        NDC x=+1 → pixel largura (extrema direita)
        NDC y=-1 → pixel altura (extremo BAIXO — Y invertido na tela!)
        NDC y=+1 → pixel 0 (extremo TOPO)

        O Y é invertido porque:
        - Em matemática: y aumenta para CIMA
        - Em telas: y aumenta para BAIXO (convenção das telas raster)

        Args:
            ndc: coordenadas NDC [x_ndc, y_ndc, ...]

        Returns:
            (px, py): coordenadas em pixels inteiros
        """
        # Mapeamento: NDC [-1,+1] → Pixel [0, largura/altura]
        px = int((ndc[0] + 1.0) * 0.5 * self.largura)
        py = int((1.0 - ndc[1]) * 0.5 * self.altura)  # Y invertido!
        return (px, py)

    # ════════════════════════════════════════════════════════════════════════
    # PASSO 7: Rasterização com Bresenham
    # ════════════════════════════════════════════════════════════════════════

    def bresenham_linha(
        self,
        x0: int, y0: int,
        x1: int, y1: int,
        cor: Tuple[int, int, int]
    ):
        """
        Algoritmo de Bresenham para desenhar uma linha entre dois pixels.

        PROBLEMA: A linha matemática entre (x0,y0) e (x1,y1) é contínua.
        Pixels são discretos. Como escolher quais colorir?

        ANALOGIA DO ERRO ACUMULADO:
        Imagine caminhar em diagonal numa grade de xadrez.
        Você sempre quer ir para frente (eixo dominante).
        O "erro" é o quanto você se desviou verticalmente da linha ideal.
        Quando o erro ultrapassa 0.5 (meio pixel), você "sobe" um pixel.

        REVOLUCIONÁRIO em 1962:
        - Sem divisões! (só multiplicações por 2 e adições)
        - Sem ponto flutuante! (só inteiros)
        - Muito rápido para hardware da época.

        ALGORITMO:
          erro ← 0
          para cada passo no eixo dominante:
            pinta o pixel atual
            erro += inclinação_normalizada
            se erro ≥ 0.5:
              avança no eixo secundário
              erro -= 1
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1  # direção em X
        sy = 1 if y0 < y1 else -1  # direção em Y

        # "Erro" inicial (multiplicado por 2 para evitar frações)
        erro = dx - dy

        # Limites da tela (evita desenhar fora)
        w, h = self.largura, self.altura

        while True:
            # Pinta o pixel se estiver dentro da tela
            if 0 <= x0 < w and 0 <= y0 < h:
                self.superficie.set_at((x0, y0), cor)

            # Chegou ao destino?
            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * erro  # erro × 2 (para eliminar o fator 0.5)

            # Decide se avança em X
            if e2 > -dy:
                erro -= dy
                x0 += sx

            # Decide se avança em Y
            if e2 < dx:
                erro += dx
                y0 += sy

    def desenhar_aresta(
        self,
        v1_clip: np.ndarray,
        v2_clip: np.ndarray,
        cor: Tuple[int, int, int]
    ):
        """
        Pipeline completo para renderizar uma aresta (aresta do wireframe).

        Executa: Clipping → Divisão-W → Viewport → Bresenham

        Args:
            v1_clip, v2_clip: vértices em coordenadas clip (após MVP)
            cor: cor RGB da aresta
        """
        # Passo 4: Recorte
        resultado = self.recortar_aresta(v1_clip, v2_clip)
        if resultado is None:
            return

        v1c, v2c = resultado

        # Passo 5: Divisão por W
        ndc1 = self.divisao_perspectiva(v1c)
        ndc2 = self.divisao_perspectiva(v2c)

        if ndc1 is None or ndc2 is None:
            return

        # Passo 6: Mapeamento de Viewport
        px1, py1 = self.ndc_para_viewport(ndc1)
        px2, py2 = self.ndc_para_viewport(ndc2)

        # Passo 7: Rasterização com Bresenham
        self.bresenham_linha(px1, py1, px2, py2, cor)

    def renderizar_objeto(
        self,
        vertices: np.ndarray,
        arestas: List[Tuple[int, int]],
        matriz_modelo: np.ndarray,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
        cor: Tuple[int, int, int] = (255, 255, 255)
    ):
        """
        Renderiza um objeto 3D completo em wireframe.

        Executa todos os 7 passos do pipeline para cada aresta do objeto.

        Args:
            vertices: coordenadas locais dos vértices (N×3)
            arestas: pares de índices de vértices [(i,j), ...]
            matriz_modelo: transformação do objeto no mundo
            view_matrix: transformação da câmera
            proj_matrix: transformação de projeção
            cor: cor RGB do wireframe
        """
        # Passos 1-3: Transforma todos os vértices pelo MVP
        clip_coords = self.transformar_vertices(
            vertices, matriz_modelo, view_matrix, proj_matrix
        )

        # Para cada aresta, executa passos 4-7
        for i, j in arestas:
            self.desenhar_aresta(clip_coords[i], clip_coords[j], cor)
