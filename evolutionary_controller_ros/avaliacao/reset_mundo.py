"""Teletransporta o robô para a pose inicial entre episódios.

Usa o serviço Ignition `/world/capture_the_flag_world/set_pose` via ros_gz_interfaces
ou subprocess para `ign service`.
"""


def resetar_robo(pose_inicial):
    raise NotImplementedError
