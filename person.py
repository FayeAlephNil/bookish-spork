from collections import defaultdict
from collections import Counter
import string

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Callable, Dict, Any, Sequence
import heapq

import matplotlib.pyplot as plt
import matplotlib.colors

class Person:
    def __init__(self, pos, identifier="", name="", region=""):
        self.pos = pos
        self.identifier = identifier
        self.name = name
        self.region = region

    def copy(self):
        return Person(self.pos, self.identifier, self.name, self.region)

    def change_name(self,new_name):
        self.name = new_name
        self.identifier = self.region + '.' + self.name

class PersonType:
    def __init__(self, generator,name="", region=""):
        self.gen = generator
        self.name = name
        self.region = region

    def identifier(self):
        return self.region + '.' + self.name
    
    def __call__(self):
        # Generate one voter
        return Person(self.gen(),self.identifier(), self.name, self.region)

    def copy(self):
        return PersonType(self.gen, self.name, self.region)

    def sample(self, n):
        # Generate n voters
        return np.array([Person(self.gen(),self.identifier(),self.name,self.region) for _ in range(n)])

    def uniform_L_infinity_ball(x,y,r=1,name="",region=""):
        # Uniform Distribution
        gen = lambda: np.random.uniform(low= (x-r,y-r),high= (x+r,y+r),size= 2)
        return PersonType(gen,name,region)

    def uniform_L1_ball(x,y,r=1, name="",region=""):
        # L1 Distribution
        def gen():
            numbers = np.random.exponential(1, size = 2) # Picks two random numbers 
            normalized_numbers = numbers / numbers.sum() # Normalize so their sum adds to one
            signs = np.random.choice([-1,1], size = 2)
            points = signs * normalized_numbers # Some point on the boundary
            radius = r * (np.random.uniform(0,1) ** (1/2))
            return np.array([x,y]) + radius * points
        return PersonType(gen,name,region)

    def uniform_L2_ball(x,y,r=1,name="",region=""):
        #L2 Distribution
        def gen():
            # Pick angle and radius on a circle of radius 1
            theta = np.random.uniform(0, 2*np.pi)
            radius = r * np.sqrt(np.random.uniform(0, 1))
            return np.array([x, y]) + rad * np.array([np.cos(theta), np.sin(theta)])
        return PersonType(gen, name,region)

    def gaussian(x, y, sigma=0.5, name="",region=""):
        # Gaussian Moment, sigma is the standard deviation 
        gen = lambda: np.random.normal(loc=(x, y), scale=sigma, size=2)
        return PersonType(gen, name,region)

    def correlated_gaussian(mean, cov, name="",region=""):
        #Correlated Gaussian with full covariance.
        mean = np.array(mean, dtype=float)
        cov = np.array(cov, dtype=float)
        gen = lambda: np.random.multivariate_normal(mean= mean, cov=cov)
        return PersonType(gen, name,region)

    def combine(voter_types,name="",region=""):
        assert sum(voter_types.values()) == 1, "Need to have proportions"
        def gen():
            my_guy = np.random.choice(voter_types.keys(), p=voter_types.values())
            return my_guy.gen()
        return PersonType(gen, name, region)

def circle_gauss_system(n,offset=0,sigma=0.5,size=1):
    centers = [(size*np.cos(2*np.pi*k/n+offset), size*np.sin(2*np.pi*k/n+offset)) for k in range(0,n)]
    people = [PersonType.gaussian(c[0],c[1],sigma=sigma,name=f"Party{k}") for k,c in enumerate(centers)]
    return people
