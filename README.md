# evolutionary_controller_ros

Controlador evolutivo para o robô diff-drive do pacote [`prm_2026`](https://github.com/matheusbg8/prm_2026), disciplina SSC0712 (Programação de Robôs Móveis — USP São Carlos, 2026). O objetivo é aprender políticas de controle via algoritmos evolutivos para resolver o cenário **Capture The Flag** definido em `prm_2026`.

## Dependências

- ROS 2 Humble
- Gazebo Fortress (`ign gazebo` 6.x)
- Pacote [`prm_2026`](https://github.com/matheusbg8/prm_2026) clonado em `src/` do mesmo workspace

## Build

```bash
cd ~/USP/Robos_moveis/ros2_ws
colcon build --symlink-install --packages-select evolutionary_controller_ros
source install/local_setup.bash
```

## Executar

Três terminais. No WSL é preciso `LIBGL_ALWAYS_SOFTWARE=1` nos que lançam Gazebo.

```bash
# 1) Mundo + bridge (pacote do professor)
LIBGL_ALWAYS_SOFTWARE=1 ros2 launch prm_2026 inicia_simulacao.launch.py

# 2) Robô + controllers + RViz (pacote do professor)
LIBGL_ALWAYS_SOFTWARE=1 ros2 launch prm_2026 carrega_robo.launch.py

# 3) Controlador evolutivo (este pacote) — em vez do controle_robo do professor
ros2 launch evolutionary_controller_ros rodar_controlador.launch.py
```

Para rodar o laço evolutivo (treinamento):
```bash
ros2 launch evolutionary_controller_ros treinar.launch.py
```

## Estrutura

```
evolutionary_controller_ros/
├── controladores/    # nós ROS2 — recebem um genoma e pilotam o robô
├── evolucao/         # núcleo do GA (puro Python, testável sem ROS)
├── avaliacao/        # orquestração de episódios, reset do mundo, fitness
└── utils/            # helpers (parsing de sensores, logging CSV)
```

A separação é deliberada: `evolucao/` não importa nada de ROS, para poder testar e iterar rápido. `controladores/` e `avaliacao/` são a interface com ROS/Gazebo.

## Licença

MIT — ver [LICENSE](LICENSE).
