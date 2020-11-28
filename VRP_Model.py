
# -*- coding: utf-8 -*-
"""
This contains the UAV class and the gurobi VRP model

Model architecture:

    K identical drones
    Time window for drone launch / land
    Capacity
    Limited endurance

Objective: minimise the total lateness and total distance

@author: danny,flori,vlad

"""

import numpy as np
import pandas as pd
from gurobipy import Model,GRB,LinExpr
from copy import deepcopy
import gurobipy as gp
import time
import os
from itertools import chain, combinations
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from statistics import mean
import matplotlib.image as mpimg
import matplotlib.text as mpl_text

from Data_prepro import getData

# consts.
M = 100000

class UAV:
    """
    The class for our drones.
    (We can add multiple modles of drones, I don't know how that will look like in the model...')
    """
    def __init__(self, maxspeed, capacity, endurance, maxrange):

        self.v = maxspeed  # [m/s]
        self.q = capacity  # [pizzaz]   1 pizza = 0.5 kg
        self.R = maxrange  # [m]
        self.E = endurance  #[s]



P,C,D,e,c,q,coord_airbase,coord_clients,coord_pizzerias,distances = getData()

print(distances)


# Buy some drones:
K       = range(3)                 # number of drones
drone   = UAV(15,10,180*60,1000)    # drone model
D       = 10                        # delay in seconds between drone launch / land

print(c)


### Create model
m = gp.Model("VRP")


#+++++++++++++++++++++++++++++++ Decision variables +++++++++++++++++++++++++++++++++++++++++++++++++

# Edge assignment to drone:
x = {}
# Add edges from airbase to pizzerias
for j in P:
    for k in K:
        x[0, j, k] = m.addVar(lb=0, ub=1, vtype=GRB.BINARY,name="x[%s,%s,%s]"%(0, j, k))
# Add edges from pizzerias to customers
for i in P:
    for j in C:
        for k in K:
            x[i, j, k] = m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name="x[%s,%s,%s]" % (i, j, k))
# Add edges from customers to customers
for i in C:
    for j in C:
        if i != j:
            for k in K:
                x[i, j, k] = m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name="x[%s,%s,%s]" % (i, j, k))
# Add edges from customers to airbase
for i in C:
    for k in K:
        x[i, 0, k] = m.addVar(lb=0, ub=1, vtype=GRB.BINARY,name="x[%s,%s,%s]"%(i, 0, k))


x_k = m.addVars(K, vtype = GRB.BINARY, name = "x_k")     #drone is active

# Arrival times for each drone, for each customer and pizzeria
tau = {}
for k in K:
    for i in P:
        tau[i,k] = m.addVar(vtype = GRB.CONTINUOUS, name = 'tau[%s,%s]'%(i, k))
    for j in C:
        tau[j,k] = m.addVar(vtype = GRB.CONTINUOUS, name = 'tau[%s,%s]'%(j, k))

# Launch times for each drone
launch = {}
for k in K:
    launch[k] = m.addVar(vtype = GRB.CONTINUOUS, name = 'launch[%s]'%(k))

# Land time for each drone
land = {}
for k in K:
    land[k] = m.addVar(vtype = GRB.CONTINUOUS, name = 'land[%s]'%(k))

# Help binary variable for either-or constraint of launch times (y) and landing times (q)
y = {}
q = {}
for combi in combinations(K, 2):
    y[combi] = m.addVar(vtype = GRB.BINARY, name = 'y[%s,%s]'%(combi[0], combi[1]))
    q[combi] = m.addVar(vtype = GRB.BINARY, name = 'q[%s,%s]'%(combi[0], combi[1]))

m.update()


#++++++++++++++++++++++++++++++++++ Constraints  ++++++++++++++++++++++++++++++++++++++++++++++++++


# 1 Leave the luanch site (index 0):
m.addConstrs((gp.quicksum(x[0,j,k] for j in P) == x_k[k]  for k in K), name = "Launch site")   # changed C to P

#2 Land at launch site (index 0):
m.addConstrs((gp.quicksum(x[i,0,k] for i in C) == x_k[k]  for k in K), name = "Landing site")

# # 3 Each pizzeria needes to be visited once:
m.addConstrs((gp.quicksum(x[0,j,k]  for k in K ) == 1 for j in P), name = "Visit pizzeria")

# 4 Each customer needes to be visited once:
m.addConstrs((gp.quicksum(x[i,j,k]  for k in K for i in chain(P, C) if i!=j) == 1 for j in C), name = "Visit customer")

# 5.1 and 5.2 Each drone must leave:
m.addConstrs((gp.quicksum(x[j,i,k] for j in chain(P, C) if i!=j) == gp.quicksum(x[i,j,k] for j in chain(C, range(1)) if i!=j) for i in C for k in K), name = "Leave customer")
m.addConstrs((gp.quicksum(x[j,i,k] for i in C) == gp.quicksum(x[i,j,k] for i in [0]) for j in P for k in K), name = "Leave pizzeria")

# 6 Lower bound on arrival time:
m.addConstrs((e[i] - tau[i,k] - (1 - x[0,i,k]) * M <= 0 for i in P for k in K), name ="time bound on pizzeria")

# 7 Time window for arriving at pizzeria:
m.addConstrs(((0 + distances[0,j]/drone.v - (1 - x[0,j,k]) * M)  <= tau[j,k] for j in P for k in K), name = "Time window pizzeria")   # the first 0 might change later if drones leave at differetn times
m.addConstrs(((tau[i,k] + distances[i,j]/drone.v - (1 - x[i,j,k]) * M) <= tau[j,k] for i in chain(P, C) for j in C for k in K if i!=j), name = "Time window pizzeria to customer and customer to customer")
# tau[j] has no upper bound except for endurance

# 8 Arrival time at customer:
m.addConstrs((c[i,0] - tau[i,k] - (1 - x[j,i,k]) * M <= 0 for i in C for j in chain(P, C) for k in K if i!=j), name = "lower bound on customer ")
m.addConstrs((c[i,1] - tau[i,k] + (1- x[j,i,k]) * M >= 0 for i in C for j in chain(P, C) for k in K if i!=j), name = "upper bound on customer ")

# 9 Launch time
m.addConstrs((gp.quicksum((tau[i,k] - distances[0,i]/drone.v)*x[0,i,k] for i in P) >= launch[k] for k in K), name = "launch time")

# 10 Landing time
m.addConstrs((gp.quicksum((tau[i,k] + distances[i,0]/drone.v)*x[i,0,k] for i in C) <= land[k] for k in K), name = "land time")

# 11 Either - or delay constraint for launch time
m.addConstrs(( launch[k] + D*x_k[m]*x_k[k] <= launch[m] + M * y[k,m] for k,m in combinations(K, 2)), name = "either launch time of drone m is D time after launch time of drone k")
m.addConstrs(( launch[k] + D*x_k[m]*x_k[k] <= launch[m] + M * (1 - y[k,m]) for k,m in combinations(K, 2)), name = "or launch time of drone k is D time after launch time of drone m")

# 12 Either - or delay constraint for land time
m.addConstrs(( land[k] + D*x_k[m]*x_k[k] <= land[m] + M * q[k,m] for k,m in combinations(K, 2)), name = "either land time of drone m is D time after land time of drone k")
m.addConstrs(( land[k] + D*x_k[m]*x_k[k] <= land[m] + M * (1 - q[k,m]) for k,m in combinations(K, 2)), name = "or land time of drone k is D time after land time of drone m")

# 13 Max endurance
m.addConstrs(( land[k] - launch[k] <= drone.E for k in K), name = "max endurance of drone")


m.update()



# ++++++++++++++++++++++++++++    Objective +++++++++++++++++++++++++++++++++++++++++++++++++++++++++



alpha     =   0   #How important are the customers?
obj = LinExpr()

for key in x:
    obj += (1-alpha)*x[key]*distances[key[0], key[1]] #part of objective function related to total distance
    if key[0] in P or key[0] in C:
        #part of objective function related to distance from pizzeria to customer and customer to customer
        obj += alpha*x[key]*distances[key[0], key[1]]
for k in K:
    obj += M*(land[k]-launch[k]) # minimise time in air, dont stay in air if unnecessary
for i in P:
    for k in K:
        obj += M*M*(tau[i,k]-e[i]) # minimise time delay vs expected arrival time at pizzeria
for j in C:
    for k in K:
        obj += M*M*(tau[j,k]-c[j,0]) # minimise time delay vs expected arrival time at pizzeria



m.setObjective(obj, GRB.MINIMIZE)
m.update()


#+++++++++++++++++++++++++++++++++++ Solve Model +++++++++++++++++++++++++++++++++++++++++++++++++++
m.write("VRP_basic.lp")


# m.computeIIS()

m.optimize()


status = m.Status