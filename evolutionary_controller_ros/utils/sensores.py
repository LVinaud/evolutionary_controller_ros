"""Helpers para transformar mensagens de sensores em features normalizadas."""


def scan_to_features(scan_msg, n_bins=16):
    raise NotImplementedError


def imagem_para_mascara_bandeira(image_msg, cor_alvo_bgr):
    raise NotImplementedError
