"""Teleports the robot back to its starting pose between episodes.

Uses the Ignition service `/world/capture_the_flag_world/set_pose` via
ros_gz_interfaces or a subprocess call to `ign service`.
"""


def reset_robot(initial_pose):
    raise NotImplementedError
