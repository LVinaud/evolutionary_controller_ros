"""Operadores evolutivos: seleção, crossover, mutação."""


class Populacao:
    def __init__(self, tamanho):
        self.tamanho = tamanho
        self.individuos = []

    def selecionar(self):
        raise NotImplementedError

    def crossover(self, pai, mae):
        raise NotImplementedError

    def mutar(self, genoma):
        raise NotImplementedError
