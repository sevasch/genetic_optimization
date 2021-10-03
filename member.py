from abc import abstractmethod
from __future__ import annotations

class Member():
    def __init__(self, chromosome, age=0):
        self.chromosome = chromosome
        self.age = age  # number of generations this member has survived

    ''' CREATION '''
    @classmethod
    @abstractmethod
    def create_random(cls) -> Member:
        ''' Randomly initialize the member. '''
        pass

    @classmethod
    def encode_from_data(cls, data) -> Member:
        ''' Create Member from phenotype data. '''
        pass

    ''' OTHER '''
    def decode(self):
        ''' Get phenotype of Member. '''
        pass

    ''' EVALUATION '''
    @abstractmethod
    def get_fitness(self, environment) -> float:
        ''' Evaluate fitness of member in a given environment. '''
        pass

    ''' GO TO NEXT GENERATION'''
    def let_survive(self):
        self.age += 1

    @abstractmethod
    def _mutate_chromosome(self) -> list:
        ''' Define method to mutate the chromosome. '''
        pass

    def mutate(self) -> Member:
        return Member(self._mutate_chromosome(), age=0)

    @abstractmethod
    def _crossover_cromosome(self, other: list) -> list:
        ''' Define method to recombine two chromosomes. '''
        pass

    def crossover(self, other: Member) -> Member:
        return Member(self._crossover_cromosome(self.chromosome, other.chromosome), age=0)