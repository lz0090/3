import numpy as np
import math as mt
import random
import matplotlib.pyplot as plt
import gym
from gym import spaces
from scipy.special import jv
from datetime import datetime

def AGV_trace(x):
    y = x**0.9+1/10*x
    return y




