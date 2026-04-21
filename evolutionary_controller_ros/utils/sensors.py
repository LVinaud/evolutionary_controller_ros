"""Helpers to turn sensor messages into normalized features."""


def scan_to_features(scan_msg, n_bins=16):
    raise NotImplementedError


def image_to_flag_mask(image_msg, target_color_bgr):
    raise NotImplementedError
