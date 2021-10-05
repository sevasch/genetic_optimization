from __future__ import annotations
from abc import abstractmethod

class Member():
    def __init__(self, chromosome, age=0):
        self.chromosome = chromosome
        self.age = age  # number of generations this member has survived

    ''' CREATION '''
    @classmethod
    @abstractmethod
    def create_random(cls, *args):
        ''' Randomly initialize the member. '''
        pass

    @classmethod
    def encode_from_data(cls, data):
        ''' Create Member from phenotype data. '''
        pass

    ''' OTHER '''
    def decode(self):
        ''' Get phenotype of Member. '''
        pass

    ''' EVALUATION '''
    @abstractmethod
    def get_fitness(self) -> float:
        ''' Evaluate fitness of member in a given environment. '''
        pass

    ''' GO TO NEXT GENERATION'''
    def let_survive(self):
        self.age += 1
        return self

    @abstractmethod
    def mutate(self) -> __class__:
        pass

    @abstractmethod
    def crossover(self, other) -> __class__:
        pass