# 🎮 Visualizador 3D — NumPy + Pygame

> Sistema completo de visualização 3D implementado **do zero** em Python puro, usando apenas NumPy para os cálculos matemáticos e Pygame para exibição. Sem OpenGL. Sem shaders. Tudo feito à mão.

---

## 📦 Instalação e Execução

```bash
# Instalar dependências
pip install numpy pygame

# Executar
python main.py
```

---

## 🕹️ Controles

| Tecla | Ação |
|-------|------|
| `W` / `S` | Mover câmera frente / trás |
| `A` / `D` | Mover câmera esquerda / direita |
| `Q` / `E` | Mover câmera baixo / cima |
| `↑↓←→` | Girar câmera (pitch / yaw) |
| `Z` / `X` | Zoom in / Zoom out (ajusta FOV) |
| `R` | Resetar câmera à posição inicial |
| `ESPAÇO` | Pausar / retomar animação |
| `Click + arrastar` | Girar câmera pelo mouse |
| `ESC` | Sair |

---

## 📁 Estrutura do Projeto

```
COMP GRAFICA/
│
├── main.py         # Loop principal: cena, animação, HUD
├── transforms.py   # Parte 1: Biblioteca de transformações 4×4
├── quaternion.py   # Parte 2: Quatérnios e SLERP
├── camera.py       # Parte 3: Câmera virtual, View & Projection Matrix
└── renderer.py     # Parte 4: Pipeline de 7 passos + Bresenham
```

---

## 📚 Teoria: As 5 Partes Explicadas

### Parte 1 — A Mágica das Coordenadas Homogêneas

**O Problema:** Matrizes 3×3 conseguem fazer rotações e escalas, mas não translações. Isso porque translação é uma *soma*, não um *produto*.

**A Solução:** Adicionamos uma 4ª coordenada falsa `w=1` a todos os pontos:  
`[x, y, z]` → `[x, y, z, 1]`

Isso converte a soma de translação numa multiplicação de matriz, permitindo combinar **todas** as transformações numa só operação:

```
[1  0  0  tx]   [x]   [x + tx]
[0  1  0  ty] × [y] = [y + ty]   ← Translação como produto!
[0  0  1  tz]   [z]   [z + tz]
[0  0  0   1]   [1]   [  1   ]
```

**Benefício prático:** Em vez de aplicar rotação, depois escala, depois translação separadamente, fazemos `MVP × vértice` — **uma única** multiplicação que faz tudo.

---

### Parte 2 — Quatérnios vs. Gimbal Lock

**O Problema (Gimbal Lock):** Euler angles (pitch/yaw/roll) representam rotações como 3 ângulos consecutivos em torno de eixos fixos. Quando dois eixos se alinham, você *perde um grau de liberdade*. O objeto não consegue girar em uma direção sem que outra se misture.

> Isso aconteceu de verdade na **missão Apollo 11** — os computadores de bordo tinham proteções especiais para evitar Gimbal Lock nos giroscópios!

**A Solução — Quatérnios:**  
`q = w + xi + yj + zk`

Um quatérnio representa qualquer rotação como **um único giro** em torno de um **eixo arbitrário**:

```
q = cos(θ/2) + sin(θ/2)·(ax·i + ay·j + az·k)
```

Como é só um giro, não há eixos aninhados, portanto não há Gimbal Lock.

**Produto de Hamilton:** Combinar duas rotações é multiplicar dois quatérnios:
```
q_combined = q1 × q2  (regras: i²=j²=k²=ijk=-1)
```

**SLERP** (Spherical Linear Interpolation): Interpola entre duas rotações *caminhando pela superfície de uma esfera* — garantindo velocidade constante e transição suave.

---

### Parte 3 — A Câmera e o Projetor

**Analogia do Projetor:**

```
[Cena 3D] → [Câmera/Slide] → [Matriz de Projeção/Lente] → [Tela]
```

**Sistema UVN** — A câmera tem sua própria base ortonormal:
- `n` = vetor "frente" (onde a câmera olha)
- `u` = vetor "direita" (`n × up`)
- `v` = vetor "cima" (`u × n`)

**View Matrix (LookAt):** Move o *mundo inteiro* para que a câmera fique na origem olhando para `-Z`. Simplifica todos os cálculos seguintes.

**Frustum e FOV:**
```
f = 1 / tan(FOV / 2)

[f/a  0    0    0  ]
[ 0   f    0    0  ]    ← a = aspect ratio
[ 0   0   (far+near)/(near-far)  2·far·near/(near-far) ]
[ 0   0   -1    0  ]
```

**Como a perspectiva surge?**  
A linha `-1` na posição `[3][2]` faz com que `w_clipe = -z_câmera`. Na divisão por `w`, objetos distantes (z grande) são divididos por um número grande → aparecem menores. **É a perspectiva matemática!**

---

### Parte 4 — O Pipeline de 7 Passos

```
Vértice Local
    ↓ [PASSO 1] Matriz de Modelo → Espaço do Mundo
    ↓ [PASSO 2] Matriz de Visão  → Espaço da Câmera
    ↓ [PASSO 3] Matriz de Proj.  → Espaço Clip (4D homogêneo)
    ↓ [PASSO 4] Clipping         → Descarta vértices fora do frustum
    ↓ [PASSO 5] Divisão por W    → NDC: coordenadas [-1, +1]
    ↓ [PASSO 6] Viewport         → Pixels da tela
    ↓ [PASSO 7] Bresenham        → Quais pixels colorir?
Pixel na Tela
```

**Algoritmo de Bresenham (1962):**  
Dado que uma linha vai de `(x0,y0)` a `(x1,y1)`:

```python
erro = dx - dy
while não chegou:
    pinta (x0, y0)
    e2 = 2 * erro
    if e2 > -dy: erro -= dy; x0 += sx   # avança em X
    if e2 < dx:  erro += dx; y0 += sy   # avança em Y
```

O "erro" mede o quanto nos desviamos da linha ideal. Quando ultrapassa o threshold, movemos para o próximo pixel no eixo secundário.

**Revolucionário porque:** usa apenas inteiros e adições — sem divisões, sem ponto flutuante. Era 10× mais rápido que alternativas na época.

---

### Parte 5 — A Cena Animada

A aplicação mostra:

1. **Cubo** rotacionando via combinação de SLERP + quatérnios (`q_spin_y × q_spin_x × q_slerp`)
2. **Pirâmide** orbitando o cubo em círculo, girando ao redor do próprio eixo
3. **Grid** de referência no plano XZ
4. **Eixos** X/Y/Z coloridos
5. **Barra SLERP** mostrando a interpolação em tempo real
6. **HUD** com posição da câmera, FOV e FPS

---

## 🔬 Conceitos Implementados

| Conceito | Arquivo | Função/Classe |
|---------:|---------|---------------|
| Coord. Homogêneas | `transforms.py` | `translacao()`, `escala()` |
| Rotações Euler | `transforms.py` | `rotacao_x/y/z()` |
| Cisalhamento | `transforms.py` | `cisalhamento_xy()` |
| Composição de matrizes | `transforms.py` | `compor()` |
| Quatérnio de Hamilton | `quaternion.py` | `class Quaternion` |
| Produto de Hamilton | `quaternion.py` | `__mul__()` |
| Conversão eixo/ângulo | `quaternion.py` | `from_axis_angle()` |
| Rotação q·p·q* | `quaternion.py` | `rotate_point()` |
| SLERP | `quaternion.py` | `slerp()` |
| Sistema UVN | `camera.py` | `_calcular_uvn()` |
| View Matrix (LookAt) | `camera.py` | `get_view_matrix()` |
| Projeção Perspectiva | `camera.py` | `get_projection_matrix()` |
| Pipeline MVP | `renderer.py` | `transformar_vertices()` |
| Clipping | `renderer.py` | `recortar_aresta()` |
| Divisão por W | `renderer.py` | `divisao_perspectiva()` |
| Mapeamento Viewport | `renderer.py` | `ndc_para_viewport()` |
| Algoritmo Bresenham | `renderer.py` | `bresenham_linha()` |
