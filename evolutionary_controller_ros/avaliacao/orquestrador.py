"""Executável ROS2: roda o laço evolutivo (população, episódios, seleção)."""
import rclpy
from rclpy.node import Node


class Orquestrador(Node):
    def __init__(self):
        super().__init__('orquestrador')
        self.declare_parameter('populacao', 50)
        self.declare_parameter('geracoes', 100)
        self.declare_parameter('taxa_mutacao', 0.1)
        self.declare_parameter('taxa_crossover', 0.7)
        self.declare_parameter('tempo_max_episodio', 60.0)
        self.declare_parameter('elite', 2)
        self.declare_parameter('seed', 42)

    def rodar(self):
        # TODO: carregar params, instanciar Populacao, iterar gerações chamando
        # rodar_episodio + operadores de evolucao/; salvar melhor em genomas/.
        raise NotImplementedError


def main():
    rclpy.init()
    node = Orquestrador()
    try:
        node.rodar()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
