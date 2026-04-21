# evolutionary_controller_ros

Controlador evolutivo para o robô diff-drive do pacote [`prm_2026`](https://github.com/matheusbg8/prm_2026), disciplina SSC0712 (Programação de Robôs Móveis — USP São Carlos, 2026).

## O que este pacote faz

O pacote `prm_2026` (do professor) traz a **plataforma**: mundo Gazebo, robô com sensores, controllers do `ros2_control`, bridges ROS↔Gazebo. Ele define o cenário **Capture The Flag** (mundo `arena_cilindros.sdf`, internamente `capture_the_flag_world`) — duas bases, duas bandeiras, zona de deploy, paredes e cilindros — mas não implementa a lógica de jogo. Entrega só stubs em `controle_robo.py` e `robo_mapper.py`.

Este pacote é o **cérebro**: implementa algoritmos evolutivos que aprendem políticas para o robô resolver o CTF (sair da base, achar bandeira adversária, pegar com o gripper, levar pra zona de deploy).

**Princípio de separação:** `prm_2026` fica intocado. Consumimos os tópicos que ele publica (`/scan`, `/imu`, `/odom_gt`, `/robot_cam/*`) e publicamos nos tópicos que ele consome (`/cmd_vel`, `/gripper_controller/commands`). Nada mais.

## Dependências

- ROS 2 Humble
- Gazebo Fortress (`ign gazebo` 6.x)
- Pacote `prm_2026` clonado em `src/` do mesmo workspace
- Python 3.10+, numpy, scipy, opencv (já vêm com ROS Humble)

## Build

```bash
cd ~/USP/Robos_moveis/ros2_ws
colcon build --symlink-install --packages-select evolutionary_controller_ros
source install/local_setup.bash
```

Com `--symlink-install` o `install/` vira ponteiro pro código em `src/` — edições não exigem rebuild, só `source` (exceto quando mudar `setup.py` ou `package.xml`).

## Executar

Três terminais. No WSL é preciso `LIBGL_ALWAYS_SOFTWARE=1` nos que lançam Gazebo (aliases `rsim` e `rrobo` no `~/.bashrc` já fazem isso).

```bash
# 1) Mundo + bridge (pacote do professor)
rsim     # = LIBGL_ALWAYS_SOFTWARE=1 ros2 launch prm_2026 inicia_simulacao.launch.py

# 2) Robô + controllers + RViz (pacote do professor)
rrobo    # = LIBGL_ALWAYS_SOFTWARE=1 ros2 launch prm_2026 carrega_robo.launch.py

# 3) Controlador evolutivo (este pacote) — em vez do controle_robo do professor
ros2 launch evolutionary_controller_ros rodar_controlador.launch.py
```

Treinamento (laço evolutivo):

```bash
ros2 launch evolutionary_controller_ros treinar.launch.py
```

Demonstração do melhor genoma treinado:

```bash
ros2 launch evolutionary_controller_ros demo_melhor.launch.py genoma:=genomas/melhor.npy
```

## Estrutura completa, arquivo por arquivo

### Raiz do pacote

| Arquivo | Função |
|---|---|
| `package.xml` | Manifesto ROS2 — nome, versão, dependências, tipo de build (`ament_python`). O `colcon` lê este arquivo. |
| `setup.py` | Setup Python + registro dos **executáveis ROS2** (seção `entry_points`). É aqui que você declara "este script vira um comando `ros2 run`". |
| `setup.cfg` | Config do setuptools — diz onde instalar os scripts. |
| `resource/evolutionary_controller_ros` | Arquivo vazio. Marker que o `ament_index` usa pra descobrir o pacote. Nunca editar. |
| `LICENSE` | MIT. |
| `README.md` | Este arquivo. |
| `.gitignore` | O que não vai pro git (ver seção "O que está no git"). |

### `launch/` — arquivos de launch

Use com `ros2 launch evolutionary_controller_ros <arquivo>`.

| Arquivo | Função |
|---|---|
| `rodar_controlador.launch.py` | Sobe só o nó `controlador_nn`. Usar quando o mundo e o robô já estão rodando nos outros terminais. |
| `treinar.launch.py` | Sobe o `orquestrador` com parâmetros de `config/ga_params.yaml`. Este é o comando de treino. |
| `demo_melhor.launch.py` | Sobe o controlador carregando um genoma específico (`genoma:=caminho/arquivo.npy`). Pra demonstrar o melhor indivíduo depois do treino. |

### `config/` — parâmetros

| Arquivo | Função |
|---|---|
| `ga_params.yaml` | Parâmetros do GA lidos pelo `orquestrador` (tamanho da pop, gerações, taxas de mutação/crossover, elite, seed). |
| `neat_config.ini` | Placeholder. Só preencher se decidirmos usar a biblioteca `neat-python`. |

### `evolutionary_controller_ros/` — código Python

O nome da pasta é igual ao do pacote (convenção `ament_python`).

```
evolutionary_controller_ros/
├── __init__.py             (vazio — marker do Python)
├── controladores/          nós ROS2 (o "corpo" que pilota o robô)
├── evolucao/               núcleo do GA (puro Python, sem ROS)
├── avaliacao/              cola entre ROS e GA
└── utils/                  helpers compartilhados
```

#### `controladores/` — as políticas que dirigem o robô

Cada arquivo aqui é um **nó ROS2 executável**. Todos subscrevem `/scan`, `/odom_gt`, `/robot_cam/colored_map` e publicam em `/cmd_vel`. A diferença entre eles é **como** decidem o `cmd_vel`.

| Arquivo | Função |
|---|---|
| `__init__.py` | Vazio. |
| `controlador_nn.py` | Rede neural. Entrada = features dos sensores. Saída = `[linear.x, angular.z]`. Pesos vêm de um genoma. |
| `controlador_reativo.py` | Campos potenciais (atrator pra bandeira + repulsor pra obstáculos). Ganhos dos campos vêm de um genoma. |

#### `evolucao/` — o algoritmo genético puro

**Esta pasta não importa nada de ROS.** É Python comum, testável com `pytest` sem Gazebo. Essa separação é o que permite iterar rápido na parte evolutiva — você testa um crossover em segundos, não precisa subir simulação.

| Arquivo | Função |
|---|---|
| `__init__.py` | Vazio. |
| `genoma.py` | Classe `Genoma` — representação (vetor de floats? grafo NEAT? árvore?) e serialização (`to_bytes`, `from_bytes`). |
| `populacao.py` | Classe `Populacao` — seleção (torneio/ranking/roleta), crossover, mutação. |
| `algoritmo.py` | `rodar_ga(populacao, avaliador, n_geracoes)` — laço principal do GA: avalia → seleciona → cruza → muta → repete. |
| `fitness.py` | `fitness_ctf(...)` — recebe dados do episódio e devolve um score. Define o que é "bom" (pegou bandeira? tempo? distância? colisões?). |

#### `avaliacao/` — onde ROS e GA conversam

| Arquivo | Função |
|---|---|
| `__init__.py` | Vazio. |
| `episodio.py` | `rodar_episodio(controlador, genoma, tempo_max)` — carrega o genoma no controlador, deixa rodar por N segundos de sim_time, coleta métricas. |
| `reset_mundo.py` | `resetar_robo(pose)` — chama o serviço Ignition pra teletransportar o robô de volta à posição inicial. Sem reset rápido, treinar é inviável. |
| `orquestrador.py` | Nó ROS2 executável (registrado como `ros2 run ... orquestrador`). Carrega params do YAML, instancia `Populacao`, roda as gerações via `rodar_ga`, salva o campeão em `genomas/`. |

#### `utils/` — helpers compartilhados

| Arquivo | Função |
|---|---|
| `__init__.py` | Vazio. |
| `sensores.py` | Converte mensagens de sensor em features normalizadas. `scan_to_features` faz bins angulares do lidar; `imagem_para_mascara_bandeira` detecta blob da cor da bandeira na câmera segmentada. |
| `logger.py` | `LoggerCSV` — registra `(geração, melhor, média)` em CSV pra plotar depois. |

### `genomas/` — artefatos de treino

Aqui ficam os genomas salvos pelo `orquestrador` durante o treino. O `.gitkeep` mantém a pasta no git mesmo vazia. O `.gitignore` exclui todo o conteúdo **exceto** `melhor.npy` — você commita só o campeão (genomas intermediários podem ser muitos MB/GB depois de várias gerações).

### `scripts/` — ferramentas fora do ROS

Scripts standalone que rodam com `python3 scripts/<nome>.py`. Não viram comandos `ros2 run`.

| Arquivo | Função |
|---|---|
| `plotar_fitness.py` | Lê CSV do `LoggerCSV` e plota a curva de evolução com matplotlib. |

### `test/` — testes unitários

Roda com `colcon test --packages-select evolutionary_controller_ros` ou direto com `pytest`.

| Arquivo | Função |
|---|---|
| `test_genoma.py` | Placeholder. Testes da classe `Genoma` virão aqui. |

## Fluxo de dados

```
Durante execução normal (controlador rodando no mundo):

  Gazebo (prm_2026)                     Este pacote
  ─────────────────                     ───────────
  /scan              ───────→  controlador ───→  /cmd_vel
  /imu               ───────→  controlador      (publicado 10 Hz)
  /odom_gt           ───────→  controlador
  /robot_cam/        ───────→  controlador
    colored_map
                                                 → /gripper_controller/commands
                                                   (quando for pegar bandeira)

Durante treinamento:

  orquestrador
     │
     │ cria
     ▼
  populacao ──→ genoma ──→ controlador (carrega os pesos)
                              │
                              │ executa 1 episódio
                              ▼
                         (robô corre no Gazebo por N segundos)
                              │
                              │ ao fim do episódio
                              ▼
  orquestrador ←── fitness ←── dados coletados
     │
     │ usa o fitness pra selecionar/cruzar/mutar
     ▼
  nova geração ...

  Entre episódios:
  orquestrador ──→ reset_mundo ──→ ign service (teleporta robô pra pose inicial)
```

## O que está no git

**Tudo rastreado**, exceto o que o `.gitignore` exclui:

| Padrão | Por que excluir |
|---|---|
| `__pycache__/`, `*.py[cod]`, `*.egg-info/` | Bytecode Python — regenerável. |
| `.pytest_cache/` | Cache do pytest — regenerável. |
| `build/`, `install/`, `log/` | Artefatos do colcon (caso você rode `colcon build` a partir da pasta do pacote por engano). Nunca commitar — são regenerados. |
| `.vscode/`, `.idea/`, `*.swp` | Arquivos de editor — pessoais, não pertencem ao repo. |
| `logs/` | Logs CSV de treino — podem crescer muito. |
| `genomas/*` exceto `.gitkeep` e `melhor.npy` | Genomas intermediários. Commita só o campeão. |

**Remote:** `git@github.com:LVinaud/evolutionary_controller_ros.git` — branch `main`.

**Primeiro commit:** `0ef606f` — scaffold inicial (os 31 arquivos descritos acima).

Pra ver o que está rastreado agora: `git ls-files` dentro da pasta do pacote.

## Convenções deste pacote

- Código em português (nomes de variáveis, comentários, docstrings) — reflete a disciplina.
- Cada executável ROS2 é um arquivo `.py` único com uma classe + função `main()`.
- `evolucao/` é puro Python, sem `import rclpy` — testável sem simulador.
- `controladores/` e `avaliacao/` falam ROS.
- Genomas são serializáveis em `.npy` (numpy) ou bytes brutos via `to_bytes`/`from_bytes`.
- Stubs estão marcados com `raise NotImplementedError` (parte evolutiva) ou comentário `# TODO:` (controladores que precisam rodar sem crashar).

## Licença

MIT — ver [LICENSE](LICENSE).
