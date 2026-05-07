"""
=============================================================================
 PARTE 1: BIBLIOTECA DE TRANSFORMAÇÕES 3D
=============================================================================

 ╔══════════════════════════════════════════════════════════════════╗
 ║  A MÁGICA DAS COORDENADAS HOMOGÊNEAS (O "1" que muda tudo)      ║
 ╚══════════════════════════════════════════════════════════════════╝

 ANALOGIA: Imagine que você quer mover uma mesa de lugar.
 Em um mundo 3D puro (matriz 3×3), você consegue:
   - GIRAR a mesa (rotação)       → multiplicação de matriz
   - ESTICAR a mesa (escala)      → multiplicação de matriz
   - INCLINAR a mesa (cisalhamento) → multiplicação de matriz

 MAS... para MOVER a mesa para outro canto da sala (translação),
 uma matriz 3×3 não consegue! A translação exige SOMA, não produto.
 Seria como tentar apertar um parafuso com uma faca — a ferramenta errada.

 SOLUÇÃO — A 4ª Dimensão Falsa (w = 1):
 Adicionamos uma 4ª coordenada 'w' aos nossos pontos: [x, y, z, 1].
 Esse '1' não representa nenhuma dimensão real — é um "truque de mágica"
 matemático que converte a SOMA de translação em uma MULTIPLICAÇÃO de matriz.

 Resultado: Uma única matriz 4×4 pode combinar TODAS as transformações:
 translação + rotação + escala + cisalhamento em UMA SÓ operação!
 
 Isso é poderoso porque, na GPU, fazer 1 multiplicação grande é muito
 mais rápido do que fazer várias operações separadas.

=============================================================================
"""

import numpy as np


def translacao(tx: float, ty: float, tz: float) -> np.ndarray:
    """
    Retorna uma matriz de translação 4×4.

    Move um objeto em tx unidades no eixo X,
    ty no eixo Y e tz no eixo Z.

    COMO FUNCIONA:
    O '1' extra (coordenada homogênea w) na última coluna é onde
    a 'mágica' acontece. Quando multiplicamos [x, y, z, 1] por essa
    matriz, os valores tx, ty, tz são automaticamente somados.

    [1  0  0  tx]   [x]   [x + tx]
    [0  1  0  ty] × [y] = [y + ty]
    [0  0  1  tz]   [z]   [z + tz]
    [0  0  0   1]   [1]   [  1   ]
    """
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0,  1]
    ], dtype=float)


def escala(sx: float, sy: float, sz: float) -> np.ndarray:
    """
    Retorna uma matriz de escala 4×4.

    Estica ou encolhe um objeto pelos fatores sx, sy, sz
    em cada respectivo eixo.

    Escala = 2.0 → dobra o tamanho
    Escala = 0.5 → reduz pela metade
    Escala = -1  → espelha (reflexão)

    [sx  0   0  0]
    [ 0  sy  0  0]
    [ 0   0 sz  0]
    [ 0   0  0  1]
    """
    return np.array([
        [sx,  0,  0, 0],
        [ 0, sy,  0, 0],
        [ 0,  0, sz, 0],
        [ 0,  0,  0, 1]
    ], dtype=float)


def rotacao_x(theta: float) -> np.ndarray:
    """
    Rotação em torno do eixo X pelo ângulo theta (em radianos).

    ANALOGIA: Imagine um espeto de churrasco no eixo X —
    os pontos giram ao redor desse espeto.
    O eixo X fica fixo; Y e Z giram entre si.

    [1    0       0    0]
    [0  cos(θ) -sin(θ) 0]
    [0  sin(θ)  cos(θ) 0]
    [0    0       0    1]
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [1, 0,  0, 0],
        [0, c, -s, 0],
        [0, s,  c, 0],
        [0, 0,  0, 1]
    ], dtype=float)


def rotacao_y(theta: float) -> np.ndarray:
    """
    Rotação em torno do eixo Y pelo ângulo theta (em radianos).

    ANALOGIA: Girar um globo terrestre — eixo Y é o polo norte/sul.
    Pontos no hemisfério mudam X e Z, mas Y permanece o mesmo.

    [ cos(θ)  0  sin(θ)  0]
    [   0     1    0     0]
    [-sin(θ)  0  cos(θ)  0]
    [   0     0    0     1]

    Note que sin(θ) está invertido em relação à rotX — isso mantém
    o sistema destro (right-handed coordinate system).
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1]
    ], dtype=float)


def rotacao_z(theta: float) -> np.ndarray:
    """
    Rotação em torno do eixo Z pelo ângulo theta (em radianos).

    ANALOGIA: Girar um relógio deitado na mesa — o ponteiro (Z)
    aponta para você enquanto X e Y giram no plano da mesa.

    [cos(θ) -sin(θ)  0  0]
    [sin(θ)  cos(θ)  0  0]
    [  0       0     1  0]
    [  0       0     0  1]
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ], dtype=float)


def cisalhamento_xy(a: float, b: float) -> np.ndarray:
    """
    Cisalhamento no plano XY (shear).

    ANALOGIA: Empurrar o topo de uma pilha de cartas para o lado —
    cada "andar" se desloca proporcionalmente ao seu Z.
    Dá aquele efeito de "paralela" ou "oblíquo".

    'a' controla quanto X é influenciado por Z.
    'b' controla quanto Y é influenciado por Z.

    [1  0  a  0]
    [0  1  b  0]
    [0  0  1  0]
    [0  0  0  1]
    """
    return np.array([
        [1, 0, a, 0],
        [0, 1, b, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=float)


def compor(*matrices: np.ndarray) -> np.ndarray:
    """
    Compõe uma sequência de transformações em uma única matriz 4×4.

    ATENÇÃO À ORDEM: A multiplicação de matrizes NÃO é comutativa!
    Primeiro girar e depois mover ≠ Primeiro mover e depois girar.

    A última matriz fornecida é aplicada PRIMEIRO ao ponto.
    Ex: compor(T, R) → aplica R depois T (T * R * ponto)

    Dica mnemônica: leia da direita para a esquerda.
    """
    resultado = np.eye(4, dtype=float)
    for m in matrices:
        resultado = resultado @ m
    return resultado
