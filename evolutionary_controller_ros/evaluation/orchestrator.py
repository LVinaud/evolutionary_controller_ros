"""ROS2 executable: runs the evolutionary loop (population, episodes, selection)."""
import rclpy
from rclpy.node import Node


class Orchestrator(Node):
    def __init__(self):
        super().__init__('orchestrator')
        self.declare_parameter('population', 50)
        self.declare_parameter('generations', 100)
        self.declare_parameter('mutation_rate', 0.1)
        self.declare_parameter('crossover_rate', 0.7)
        self.declare_parameter('max_episode_time', 60.0)
        self.declare_parameter('elite', 2)
        self.declare_parameter('seed', 42)

    def run(self):
        # TODO: load params, instantiate Population, iterate generations calling
        # run_episode + operators from evolution/; save best into genomes/.
        raise NotImplementedError


def main():
    rclpy.init()
    node = Orchestrator()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
