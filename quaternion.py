"""
=============================================================================
 PARTE 2: CLASSE QUATERNION (ROTAÇÕES SUAVES SEM GIMBAL LOCK)
=============================================================================

 ╔══════════════════════════════════════════════════════════════════╗
 ║  POR QUE QUATÉRNIOS? O PROBLEMA DO GIMBAL LOCK                  ║
 ╚══════════════════════════════════════════════════════════════════╝

 ANALOGIA DO GIMBAL LOCK:
 Imagine um avião com 3 giroscópios (um para cada eixo: pitch, yaw, roll).
 Cada giroscópio está montado DENTRO do anterior, como bonecas russas.

 O problema: se você inclinar o avião 90° para cima (pitch),
 o giroscópio de yaw e o de roll ficam ALINHADOS no mesmo plano.
 Você PERDE um grau de liberdade — o avião não consegue mais girar
 em uma direção sem que vire em outra ao mesmo tempo.
 
 Isso literalmente aconteceu na missão Apollo 11!
 Os astronautas tinham que manobrar para não entrar em situação de Gimbal Lock.

 SOLUÇÃO — OS QUATÉRNIOS:
 Em vez de 3 ângulos separados (Euler), um quatérnio representa a rotação
 como UM ÚNICO GIRO em torno de um eixo arbitrário:
   q = w + xi + yj + zk
 onde w é "quanto girou" e (x,y,z) é "em torno de qual eixo".

 Porque é só um giro, não há sequência de eixos aninhados,
 portanto não há Gimbal Lock. Bônus: interpolação suave com SLERP!

 ESTRUTURA MATEMÁTICA:
   i² = j² = k² = ijk = -1  (as "regras" de Hamilton, 1843)
   
=============================================================================
"""

import numpy as np
from typing import Union


class Quaternion:
    """
    Representa uma rotação 3D usando um quatérnio de Hamilton.

    Forma: q = w + xi + yj + zk
    onde w é o componente escalar (real) e (x,y,z) é o vetor.

    Para rotações puras: ||q|| = 1 (quatérnio unitário).
    """

    def __init__(self, w: float, x: float, y: float, z: float):
        """
        Cria um quatérnio q = w + xi + yj + zk.

        Args:
            w: componente escalar (cosseno da metade do ângulo)
            x, y, z: componentes vetoriais (eixo de rotação × seno da metade do ângulo)
        """
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    # ── Propriedades básicas ─────────────────────────────────────────────────

    def norm(self) -> float:
        """
        Calcula a norma (comprimento) do quatérnio.

        Para rotações, queremos ||q|| = 1 (quatérnio unitário).
        Se não for 1, a rotação vai escalar o objeto indesejavelmente.
        """
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> "Quaternion":
        """
        Retorna o quatérnio normalizado (comprimento = 1).

        É como "apontar" um vetor sem se importar com seu tamanho —
        só a direção (= eixo de rotação) importa.
        """
        n = self.norm()
        if n < 1e-10:
            return Quaternion(1, 0, 0, 0)  # Identidade se norma ~= 0
        return Quaternion(self.w / n, self.x / n, self.y / n, self.z / n)

    def conjugate(self) -> "Quaternion":
        """
        Retorna o conjugado: q* = w - xi - yj - zk.

        ANALOGIA: é o "espelho" do quatérnio — inverte a direção da rotação.
        Para um quatérnio unitário, conjugado = inverso (rotação ao contrário).
        """
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    # ── Produto de Hamilton ──────────────────────────────────────────────────

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        """
        Produto de Hamilton entre dois quatérnios: q1 × q2.

        ANALOGIA: assim como girar um objeto A e depois girar B
        produz uma rotação combinada C, multiplicar dois quatérnios
        combina duas rotações em UMA ÚNICA rotação equivalente.

        Regras de Hamilton: i²=j²=k²=-1, ij=k, jk=i, ki=j, ji=-k...

        Fórmula expandida:
          w = w1w2 - x1x2 - y1y2 - z1z2
          x = w1x2 + x1w2 + y1z2 - z1y2
          y = w1y2 - x1z2 + y1w2 + z1x2
          z = w1z2 + x1y2 - y1x2 + z1w2

        ATENÇÃO: NÃO comutativo! q1*q2 ≠ q2*q1
        (girar X depois Y ≠ girar Y depois X)
        """
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z

        return Quaternion(
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        )

    # ── Conversões ───────────────────────────────────────────────────────────

    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, angle: float) -> "Quaternion":
        """
        Cria um quatérnio a partir de um eixo de rotação e um ângulo.

        ANALOGIA: Imagine um prego cravado em um ponto —
        'axis' é a direção do prego (eixo) e 'angle' é quanto girar em torno dele.

        Fórmula:
          q = cos(θ/2) + sin(θ/2) * (ax·i + ay·j + az·k)

        O ângulo é dividido por 2 por uma propriedade geométrica dos quatérnios
        (eles "perambulam" em um espaço de dimensão dupla — a esfera S³).

        Args:
            axis: vetor 3D unitário indicando o eixo de rotação
            angle: ângulo em radianos
        """
        axis = np.array(axis, dtype=float)
        # Garante que o eixo seja unitário
        norma = np.linalg.norm(axis)
        if norma < 1e-10:
            return cls(1, 0, 0, 0)  # Sem rotação se eixo nulo
        axis = axis / norma

        half = angle / 2.0
        s = np.sin(half)
        return cls(
            w=np.cos(half),
            x=axis[0] * s,
            y=axis[1] * s,
            z=axis[2] * s
        )

    def to_matrix(self) -> np.ndarray:
        """
        Converte o quatérnio para uma matriz de rotação 4×4.

        Útil para integrar com o pipeline de transformações.
        A fórmula expande o produto q·p·q* numa forma matricial equivalente.
        """
        q = self.normalize()
        w, x, y, z = q.w, q.x, q.y, q.z

        return np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w), 0],
            [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w), 0],
            [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y), 0],
            [                0,                 0,                  0, 1]
        ], dtype=float)

    def rotate_point(self, point: np.ndarray) -> np.ndarray:
        """
        Aplica a rotação do quatérnio a um ponto 3D.

        Usa a sandwiche de Hamilton: p' = q · p · q*
        onde p é o ponto representado como quatérnio puro (w=0).

        ANALOGIA: O ponto é um "sanduíche" entre q e seu conjugado q*.
        É como colocar o ponto no meio de dois espelhos opostos —
        o resultado é a rotação pura sem reflexão ou escala.

        Args:
            point: array [x, y, z]

        Returns:
            ponto rotacionado como array [x, y, z]
        """
        # Ponto como quatérnio puro
        p = Quaternion(0, point[0], point[1], point[2])

        # Aplica rotação: p' = q × p × q*
        q_norm = self.normalize()
        p_rotado = q_norm * p * q_norm.conjugate()

        return np.array([p_rotado.x, p_rotado.y, p_rotado.z])

    def __repr__(self) -> str:
        return f"Quaternion(w={self.w:.4f}, x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"


# ── Funções de interpolação ──────────────────────────────────────────────────

def slerp(q1: Quaternion, q2: Quaternion, t: float) -> Quaternion:
    """
    SLERP: Spherical Linear Interpolation (Interpolação Linear Esférica).

    ANALOGIA: Pense em dois pontos na superfície de uma bola de futebol.
    A interpolação LINEAR normal cortaria pelo interior da bola (atalho).
    O SLERP "caminha pela superfície" da esfera, garantindo velocidade constante.

    Para rotações, isso significa que a transição parece natural e suave,
    sem aceleração ou desaceleração estranha no meio do caminho.

    Args:
        q1: rotação inicial
        q2: rotação final
        t: parâmetro de interpolação [0, 1]
           t=0 → q1, t=1 → q2, t=0.5 → meio do caminho

    Returns:
        Quaternion interpolado
    """
    # Normaliza ambos
    q1 = q1.normalize()
    q2 = q2.normalize()

    # Produto interno (cosseno do ângulo entre eles na hiperesfera)
    dot = q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z

    # Se o produto interno é negativo, q2 está no "hemisfério oposto".
    # Invertemos q2 para tomar o caminho mais curto na esfera.
    if dot < 0.0:
        q2 = Quaternion(-q2.w, -q2.x, -q2.y, -q2.z)
        dot = -dot

    # Clamp para evitar erros numéricos no arccos
    dot = min(dot, 1.0)

    # Se muito próximos, usa interpolação linear (evita divisão por zero)
    if dot > 0.9995:
        resultado = Quaternion(
            q1.w + t * (q2.w - q1.w),
            q1.x + t * (q2.x - q1.x),
            q1.y + t * (q2.y - q1.y),
            q1.z + t * (q2.z - q1.z)
        )
        return resultado.normalize()

    # SLERP real: interpola ao longo do arco da esfera
    theta_0 = np.arccos(dot)          # ângulo total entre q1 e q2
    theta = theta_0 * t               # ângulo percorrido até t

    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0

    return Quaternion(
        s1 * q1.w + s2 * q2.w,
        s1 * q1.x + s2 * q2.x,
        s1 * q1.y + s2 * q2.y,
        s1 * q1.z + s2 * q2.z
    ).normalize()
