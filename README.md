# 🎮 Visualizador 3D — NumPy + Pygame

> Sistema completo de visualização 3D implementado **do zero** em Python puro, usando apenas NumPy para os cálculos matemáticos e Pygame para exibição. Sem OpenGL. Sem shaders. Tudo feito à mão.

---

## 📦 Instalação e Execução

```bash
# Instalar dependências
pip install numpy pygame-ce

# Executar
python main.py
```

> **Requisitos:** Python 3.10+ | NumPy | pygame-ce 2.5+

---

## 🖥️ Telas da Aplicação

A aplicação possui **3 telas** principais:

| Tela | Descrição |
|------|-----------|
| **Menu** | Lista interativa de sólidos + Cena Demo, com painel de descrição |
| **Visualizador Individual** | Sólido 3D com rotação automática, câmera livre e **modo manual** |
| **Cena Demo (Parte 5)** | Cubo + pirâmide orbitando + demonstração de SLERP |

---

## 🕹️ Controles

### Menu

| Tecla | Ação |
|-------|------|
| `↑` / `↓` ou `W` / `S` | Navegar entre os sólidos |
| `ENTER` ou `ESPAÇO` | Entrar no visualizador selecionado |
| Click no card | Selecionar sólido (duplo-click para entrar) |
| `ESC` | Sair da aplicação |

### Visualizador — Modo AUTO (padrão)

O sólido rotaciona automaticamente via quaternions com SLERP oscilando.

| Tecla | Ação |
|-------|------|
| `W` / `S` | Mover câmera frente / trás |
| `A` / `D` | Mover câmera esquerda / direita |
| `Q` / `E` | Mover câmera baixo / cima |
| `↑` `↓` `←` `→` | Girar câmera (pitch / yaw) |
| `Z` / `X` | Zoom in / out (ajusta FOV) |
| `R` | Resetar câmera à posição inicial |
| `ESPAÇO` | Pausar / retomar animação automática |
| Click + arrastar | Girar câmera pelo mouse |
| `TAB` | **Alternar para modo MANUAL** |
| `ESC` | Voltar ao menu |

### Visualizador — Modo MANUAL (interativo)

O usuário controla **manualmente cada transformação** aplicada ao objeto.

| Tecla / Ação | Função |
|--------------|--------|
| `TAB` | Voltar ao modo AUTO |
| `1` - `7` | Selecionar transformação ativa por tecla |
| Click no card | Selecionar transformação ativa por mouse |
| Mouse drag (arraste) | Ajustar parâmetros da transformação ativa |
| Scroll wheel | Ajuste fino (eixo Z, escala, ângulo) |
| `BACKSPACE` | Resetar transformação selecionada ao padrão |
| `↑` `↓` `←` `→` | Girar câmera (mantido) |
| `R` | Resetar câmera |
| `ESC` | Voltar ao menu |

#### Transformações Disponíveis no Modo Manual

| # | Transformação | Mouse Drag | Scroll Wheel | Função Interna |
|---|---------------|-----------|-------------|----------------|
| 1 | **Translação** | X → TX, Y → TY | TZ | `translacao(tx, ty, tz)` |
| 2 | **Escala** (uniforme) | Y → fator | fator | `escala(s, s, s)` |
| 3 | **Rotação X** (Euler) | Y → ângulo | ângulo | `rotacao_x(θ)` |
| 4 | **Rotação Y** (Euler) | X → ângulo | ângulo | `rotacao_y(θ)` |
| 5 | **Rotação Z** (Euler) | X → ângulo | ângulo | `rotacao_z(θ)` |
| 6 | **SLERP** (quaternion) | X → t [0, 1] | t | `slerp(qA, qB, t)` |
| 7 | **Cisalhamento** (shear XY) | X → a, Y → b | a | `cisalhamento_xy(a, b)` |

> **Composição de ordem fixa:** Translação ← SLERP ← RotZ ← RotY ← RotX ← Escala ← Cisalhamento ← Escala Base

### Cena Demo (Parte 5)

| Tecla | Ação |
|-------|------|
| `W/S/A/D/Q/E` | Mover câmera |
| `↑` `↓` `←` `→` | Girar câmera |
| `Z` / `X` | Zoom |
| Click + arrastar | Girar câmera pelo mouse |
| `ESPAÇO` | Pausar / retomar animação |
| `R` | Resetar câmera |
| `ESC` | Voltar ao menu |

---

## 📁 Estrutura do Projeto

```
projetoPython/
│
├── main.py                    # Ponto de entrada: Menu + Visualizador + CenaDemo
├── requirements.txt           # Dependências (numpy, pygame-ce)
├── .gitignore                 # Arquivos ignorados pelo Git
│
├── engine/                    # Pacote do motor 3D
│   ├── __init__.py            # Exporta a API pública do engine
│   ├── transforms.py          # Parte 1: Biblioteca de transformações 4×4
│   ├── quaternion.py          # Parte 2: Quatérnios de Hamilton + SLERP
│   ├── camera.py              # Parte 3: Câmera virtual + projeção perspectiva
│   ├── renderer.py            # Parte 4: Pipeline de 7 passos + Bresenham + Phong
│   └── mesh.py                # Objetos 3D (Objeto3D + Mesh) + catálogo de sólidos
│
├── docs/                      # Documentação e imagens do enunciado
│   └── *.jpeg                 # Páginas do enunciado do trabalho
│
└── README.md                  # Este arquivo
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

**Transformações implementadas:**

| Transformação | Função | Descrição |
|---------------|--------|-----------|
| Translação | `translacao(tx, ty, tz)` | Move o objeto nos 3 eixos |
| Escala | `escala(sx, sy, sz)` | Aumenta/diminui/espelha em cada eixo |
| Rotação X | `rotacao_x(θ)` | Gira em torno do eixo X (regra mão-direita) |
| Rotação Y | `rotacao_y(θ)` | Gira em torno do eixo Y |
| Rotação Z | `rotacao_z(θ)` | Gira em torno do eixo Z |
| Cisalhamento | `cisalhamento_xy(a, b)` | Deforma no plano XY proporcional a Z |
| Composição | `compor(*matrizes)` | Multiplica N matrizes em ordem (direita → esquerda) |

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

**API implementada:**

| Operação | Método | Descrição |
|----------|--------|-----------|
| Construtor | `Quaternion(w, x, y, z)` | Cria quatérnio de Hamilton |
| De eixo/ângulo | `Quaternion.from_axis_angle(eixo, θ)` | Cria rotação a partir de eixo e ângulo |
| Produto | `q1 * q2` | Composição de rotações (não comutativo!) |
| Para matriz | `q.to_matrix()` / `q.to_rotation_matrix()` | Converte para matriz 4×4 |
| Rotacionar ponto | `q.rotate_point(p)` | Aplica a rotação sandwiche q·p·q* |
| Normalizar | `q.normalize()` | Garante ‖q‖ = 1 (quatérnio unitário) |
| Conjugado | `q.conjugate()` | Inverte a rotação: q* = w − xi − yj − zk |
| SLERP | `slerp(q1, q2, t)` | Interpolação esférica entre q1 e q2 |

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

**Movimentação implementada:**

| Método | Ação |
|--------|------|
| `mover_frente(v)` / `mover_tras(v)` | Avança/recua na direção do olhar |
| `mover_direita(v)` / `mover_esquerda(v)` | Move lateralmente |
| `mover_cima(v)` / `mover_baixo(v)` | Move verticalmente (eixo Y do mundo) |
| `rotacionar(Δyaw, Δpitch)` | Girar olhar (com limite de ±85° no pitch) |
| `zoom(Δfov)` | Ajusta FOV entre 10° e 120° |

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
    ↓ [PASSO 7] Rasterização     → Quais pixels colorir?
Pixel na Tela
```

**Dois modos de rasterização:**

| Modo | Descrição | Uso |
|------|-----------|-----|
| **Wireframe** | Arestas via Bresenham | Grid de chão, linhas auxiliares |
| **Sólido** | Faces preenchidas + Blinn-Phong | Sólidos geométricos |

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

**Renderização sólida (Algoritmo do Pintor + Blinn-Phong):**

1. Calcula a **normal** de cada face em espaço de mundo (Newell simplificado)
2. **Backface Culling** — descarta faces viradas de costas para a câmera (dot(N, viewDir) ≥ 0)
3. **Algoritmo do Pintor** — ordena faces do mais distante ao mais próximo em Z de view
4. **Iluminação Blinn-Phong** por face (flat shading):
   - Luz principal (key light) → difusa + especular
   - Luz de preenchimento (fill light) → suaviza sombras
   - Rim light (efeito Fresnel) → brilho nas bordas
   - **I = Ka + Kd·max(0, N·L) + Ks·max(0, N·H)^48 + fill + rim**
5. Desenha **bordas anti-aliased** sobre as faces para definição

---

### Parte 5 — A Cena Animada (0,5 ponto extra)

A cena demonstra a composição de múltiplas transformações em objetos simultâneos:

1. **Cubo** rotacionando via combinação de SLERP + produto de quatérnions
2. **Pirâmide** orbitando o cubo em trajetória circular (translação + rotação própria)
3. **Grid** de referência no plano XZ (wireframe via Bresenham)
4. **Barra SLERP** mostrando o parâmetro `t` em tempo real
5. **HUD** com posição da câmera, FOV, FPS e controles

---

## 🧪 Modo Manual Interativo

O **modo manual** (ativado com `TAB` no Visualizador Individual) permite testar **todas as 7 transformações geométricas** do engine de forma interativa:

```
┌──────────────────────────────────┐
│  MODO: MANUAL                     │
├──────────────────────────────────┤
│ 1 Translação    (+0.50, -0.20, 0.00) │
│ 2 Escala        1.30x                │
│ 3 Rotação X     +45.0°          [←]│
│ 4 Rotação Y     +0.0°               │
│ 5 Rotação Z     +0.0°               │
│ 6 SLERP         t = 0.500           │
│ 7 Cisalhamento  a=+0.00  b=+0.00    │
└──────────────────────────────────┘
```

- Cada card é **clicável** e mostra o valor atual em tempo real
- O mouse controla o **objeto** (arraste para ajustar, scroll para eixo Z/fino)
- As setas continuam controlando a **câmera**
- `BACKSPACE` reseta a transformação selecionada
- A composição de todas as transformações é aplicada via `compor()` a cada frame

> **Nota:** O modo manual está disponível apenas nos visualizadores individuais. A Cena Demo (Parte 5) mantém seu funcionamento automático.

---

## 🔷 Catálogo de Sólidos (8 objetos)

| # | Sólido | Vértices | Faces | Arestas | Tipo |
|---|--------|:--------:|:-----:|:-------:|------|
| 1 | **Cubo** (Hexaedro) | 8 | 6 | 12 | Sólido de Platão |
| 2 | **Tetraedro** | 4 | 4 | 6 | Sólido de Platão |
| 3 | **Octaedro** | 6 | 8 | 12 | Sólido de Platão |
| 4 | **Icosaedro** | 12 | 20 | 30 | Sólido de Platão |
| 5 | **Pirâmide** | 5 | 5 | 8 | Pirâmide Quadrada |
| 6 | **Prisma Triangular** | 6 | 5 | 9 | Prisma |
| 7 | **Esfera** (UV) | 408 | 384 | ~ | Quádrica |
| 8 | **Toro** (Rosca) | 504 | 504 | ~ | Superfície de Revolução |

---

## 🎨 Interface Visual

### Menu
- Fundo gradiente animado com **sistema de partículas** (70 partículas)
- Título com **pulso de cor** (animação ciano ↔ roxo)
- Cards de sólidos com **borda animada** ao selecionar
- Painel de descrição com propriedades geométricas
- Barra de cor por sólido e botão ENTER pulsante

### Visualizador
- Fundo gradiente escuro (dark mode)
- **Grid de chão** no plano XZ (wireframe Bresenham)
- **HUD** com informações em painéis translúcidos:
  - Painel esquerdo: nome, vértices, faces, arestas, tipo, FPS
  - Painel direito: controles contextuais (mudam entre AUTO/MANUAL)
  - Painel câmera: posição, FOV, yaw, pitch
  - Barra SLERP: progresso `t` em tempo real
  - Badge de modo: `MODO: AUTO` (ciano) / `MODO: MANUAL` (roxo)
- **Cards clicáveis** de transformações no modo manual

### Cena Demo
- Cubo + pirâmide orbitando com **trajetória circular**
- Flutuação vertical senoidal da pirâmide
- Grid de chão + barra SLERP
- HUD com informações de órbita

---

## 🔬 Conceitos Implementados (Referência Completa)

| Conceito | Arquivo | Função/Classe |
|---------:|---------|---------------|
| Coord. Homogêneas | `engine/transforms.py` | `translacao()`, `escala()` |
| Rotações Euler (X, Y, Z) | `engine/transforms.py` | `rotacao_x()`, `rotacao_y()`, `rotacao_z()` |
| Cisalhamento (Shear) | `engine/transforms.py` | `cisalhamento_xy()` |
| Composição de matrizes | `engine/transforms.py` | `compor(*matrices)` |
| Quatérnio de Hamilton | `engine/quaternion.py` | `class Quaternion` |
| Produto de Hamilton | `engine/quaternion.py` | `Quaternion.__mul__()` |
| Conversão eixo/ângulo → quaternion | `engine/quaternion.py` | `Quaternion.from_axis_angle()` |
| Conversão quaternion → matriz 4×4 | `engine/quaternion.py` | `Quaternion.to_matrix()` |
| Rotação sandwiche q·p·q* | `engine/quaternion.py` | `Quaternion.rotate_point()` |
| Normalização de quaternion | `engine/quaternion.py` | `Quaternion.normalize()` |
| Conjugado de quaternion | `engine/quaternion.py` | `Quaternion.conjugate()` |
| SLERP | `engine/quaternion.py` | `slerp(q1, q2, t)` |
| Sistema UVN (base da câmera) | `engine/camera.py` | `Camera._calcular_uvn()` |
| View Matrix (LookAt) | `engine/camera.py` | `Camera.get_view_matrix()` |
| Projeção Perspectiva (Frustum) | `engine/camera.py` | `Camera.get_projection_matrix()` |
| Movimentação de câmera (6 DOF) | `engine/camera.py` | `mover_frente/tras/esquerda/direita/cima/baixo()` |
| Rotação de câmera (yaw/pitch) | `engine/camera.py` | `Camera.rotacionar()` |
| Zoom (FOV) | `engine/camera.py` | `Camera.zoom()` |
| Classe Objeto3D (enunciado) | `engine/mesh.py` | `class Objeto3D` |
| Classe Mesh (catálogo) | `engine/mesh.py` | `class Mesh` |
| Geração de sólidos de Platão | `engine/mesh.py` | `criar_cubo/tetraedro/octaedro/icosaedro()` |
| Geração de superfícies de revolução | `engine/mesh.py` | `criar_esfera()`, `criar_toro()` |
| Geração de pirâmide e prisma | `engine/mesh.py` | `criar_piramide()`, `criar_prisma_triangular()` |
| Pipeline MVP completo | `engine/renderer.py` | `Renderizador.render()` |
| Transformação de vértices (homogênea) | `engine/renderer.py` | `Renderizador._transform_vertices()` |
| Clipping + Divisão por W + Viewport | `engine/renderer.py` | `Renderizador._clip_to_ndc_screen()` |
| Algoritmo de Bresenham (wireframe) | `engine/renderer.py` | `Renderizador.bresenham_linha()` |
| Renderização wireframe | `engine/renderer.py` | `Renderizador.renderizar_wireframe()` |
| Renderização sólida (Painter) | `engine/renderer.py` | `Renderizador.renderizar_solido()` |
| Normal de face (Newell) | `engine/renderer.py` | `Renderizador._face_normal()` |
| Backface Culling | `engine/renderer.py` | dentro de `renderizar_solido()` |
| Iluminação Blinn-Phong | `engine/renderer.py` | dentro de `renderizar_solido()` |
| Fill Light + Rim Light (Fresnel) | `engine/renderer.py` | dentro de `renderizar_solido()` |
| Menu interativo com cards | `main.py` | `class Menu` |
| Sistema de partículas (fundo) | `main.py` | `class Particle` |
| Visualizador individual (auto/manual) | `main.py` | `class Visualizador` |
| Cena animada (Parte 5) | `main.py` | `class CenaDemo` |
| Modo manual (7 transformações) | `main.py` | `_draw_hud_manual()`, helpers |
| Redimensionamento dinâmico (resize) | `main.py` | `VIDEORESIZE` no loop principal |

---

## 🧠 Decisões de Projeto

- **Sem OpenGL/shaders:** Todo o pipeline é feito em software para fins didáticos
- **Flat shading:** Iluminação calculada por face (não por vértice), mantendo simplicidade
- **Algoritmo do Pintor:** Ordenação de faces por profundidade (funciona bem para convexos)
- **Quatérnios para rotação automática:** Evita gimbal lock na demo contínua
- **Euler angles para modo manual:** Mais intuitivo para controle interativo por eixo
- **Escala uniforme no modo manual:** Evita deformações não intencionais
- **Mouse controla objeto no manual, câmera no auto:** Evita conflito de controles
- **Setas sempre na câmera:** Padrão consistente em ambos os modos
