# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 17:47:35 2020


The purpose is (well) to do the sensitivity analysis.


@author: vladg,danny,flori
"""

import numpy as np
from itertools import chain, combinations

import pandas as pd
from gurobipy import Model,GRB,LinExpr
from copy import deepcopy
import gurobipy as gp
import time
from itertools import chain, combinations
from statistics import mean

from VRP_Model import *
