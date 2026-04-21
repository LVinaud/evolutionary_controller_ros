"""Genome representation and serialization (load/save)."""


class Genome:
    def __init__(self):
        raise NotImplementedError

    def to_bytes(self):
        raise NotImplementedError

    @classmethod
    def from_bytes(cls, data):
        raise NotImplementedError
