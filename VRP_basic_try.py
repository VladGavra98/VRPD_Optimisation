# -*- coding: utf-8 -*-
"""
Just trying around the (basic) set of contrsints and some objective.

Model architecture:

    K drones
    Time window/delay for drone launch / land
    Capacity
    Limited endurance

Objective: minimise the total lateness?

@author: vladg

"""
import numpy as np
import pandas as pd
from gurobipy import Model,GRB,LinExpr
from copy import deepcopy
import gurobipy as gp
import time
import os



M = 10000

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


# Load data:
data    = np.genfromtxt("dummy_data.csv",skip_header=1,delimiter=',',dtype=int)

# Data format: node1, node2, distance


# Nodes (except deport) & Links:
N       = int(max(max(data[:,0]),max(data[:,1])))     #nodes = locations
L       = int(len(data[:,0]))                         # links


# Pizzeria:
Pmax = 3
P       = range(Pmax)

# Arrival times:
e       = range(len(P))

# Customers:
Cmax = 5
C       = range(Cmax)

# Destinations := P + C
D       = range(N)

# Buy some drones:
K       = range(2)               # number of drones
drone   = UAV(1,1,30*60,1000)    # drone model


# custumer order:
q = [1,1,1,1,1]


# ++++++++++++++++++++++ Build data strcutures  +++++++++++++++++++++++++++++++++++++++++++++++++++

# 1.  airbase - pizzerias   (directed)   <var_name>_AP
# 2.  pizzerias - customers (directed)   <var_name>_PC
# 3.  customer - customer   (UNdirected) <var_name>_CC
# 4.  customer - airbase    (directed)   <var_name>_CA

# Build the graph as a list of tuples:



links = gp.tuplelist()
cost  = {}
distances = np.zeros((N+1,N+1))



for i in range(L):

    from_node = data[i,0]
    to_node   = data[i,1]
    cost_arc  = data[i,2]

    links.append((from_node,to_node))

    distances[from_node, to_node] = cost_arc

# # Build useful data structures
# J = [j.loc for j in customers]
# L = list(set([l[0] for l in dist.keys()]))
# D = list(set([t.depot for t in technicians]))
# cap = {k.name : k.cap for k in technicians}
# loc = {j.name : j.loc for j in customers}
# depot = {k.name : k.depot for k in technicians}
# canCover = {j.name : [k.name for k in j.job.coveredBy] for j in customers}
# dur = {j.name : j.job.duration for j in customers}
# tStart = {j.name : j.tStart for j in customers}
# tEnd = {j.name : j.tEnd for j in customers}
# tDue = {j.name : j.tDue for j in customers}


### Create model
m = gp.Model("VRP")

#+++++++++++++++++++++++++++++ Decision variables +++++++++++++++++++++++++++++++++++++++++++++++++

# Edge assignment to drone:
x   = m.addVars(L, L, K, vtype = GRB.BINARY, name="x")   #link active
x_k = m.addVars(K, vtype = GRB.BINARY, name = "x_k")     #drone is active
tau = m.addVars(P, vtype = GRB.INTEGER, name = 'tau')    #real arrival time at pizzeria


# Start time of service
t = m.addVars(L, ub=600, name="t")

# Lateness of service
z = m.addVars(C, name="z")

# Artificial variables to correct time window upper and lower limits
xa = m.addVars(C, name="xa")
xb = m.addVars(C, name="xb")


#+++++++++++++++++++++++++++++++++ Constraints  ++++++++++++++++++++++++++++++++++++++++++++++++++

# 1 All must leave the luanch site (index 0):
m.addConstrs((gp.quicksum(x[0,j,k] for j in C) == x_k[k]  for k in K), name = "Launch site")

# 2 All must leave the landing site (index 0):
m.addConstrs((gp.quicksum(x[i,0,k] for i in C) == x_k[k]  for k in K), name = "Landing site")

# 3 Each pizzeria needes to be visited once:
m.addConstrs((gp.quicksum(x[i,j,k]  for k in K for j in C)== 1 for i in P), name="visited pizzeria")

# 4 Each pizzeria needes to be visited once:
m.addConstrs((gp.quicksum(x[i,j,k]  for k in K for i in P)== 1 for j in C), name="visited customer")

# 5.1 and 5.2 Each drone must leave:
m.addConstrs((gp.quicksum(x[j,i,k] for j in C) == gp.quicksum(x[i,j,k] for j in C) for i in C for k in K), name="leave customer")
m.addConstrs((gp.quicksum(x[j,i,k] for i in C) == gp.quicksum(x[i,j,k] for i in range(0)) for j in P for k in K), name="leave pizzeria")

# 6 Lower bound on arrival time:
m.addConstrs((e[i] <=  tau[i] for i in P), name="time bound on pizzeria")

# 7 Time window for arriving at pizzeria:
m.addConstrs((0 + distances[0,j]/drone.v + (1- x[i,j,k]) * M)  <= tau[j] for j in P for k in K)   # the first 0 might change later if drones leave at differetn times

# # ?? Time window for arriving at customer:
# m.addConstrs((0 + distance[0,j]/drone.v + (1- x[i,j,k]) * M)  <= tau[j] for j in P for k in K)   # the first 0 might change later if drones leave at differetn times

# 8 Capacity constraints
m.addConstrs((gp.quicksum(q[j] * x[i,j,k] for i in C for j in C if i!=j) + (gp.quicksum(q[j] * x[i,j,k] for i in P for j in C if i!=j) )<= drone.q for k in K), name="Capacity")



# # Temporal constraints (8) for customer locations
# M = {(i,j) : 600 + dur[i] + dist[loc[i], loc[j]] for i in C for j in C}
# m.addConstrs((t[loc[j]] >= t[loc[i]] + dur[i] + dist[loc[i], loc[j]]\
#     - M[i,j]*(1 - gp.quicksum(y[loc[i],loc[j],k] for k in K))\
#     for i in C for j in C), name="tempoCustomer")

# # Temporal constraints (8) for depot locations
# M = {(i,j) : 600 + dist[i, loc[j]] for i in D for j in C}
# m.addConstrs((t[loc[j]] >= t[i] + dist[i, loc[j]]\
#     - M[i,j]*(1 - y.sum(i,loc[j],'*')) for i in D for j in C),\
#     name="tempoDepot")



# # Lateness constraint (11)
# m.addConstrs((z[j] >= t[loc[j]] + dur[j] - tDue[j] for j in C),\
#     name="lateness")

# ++++++++++++++++++++++++++++ Objective +++++++++++++++++++++++++++++++++++++++++++++++++++++++++



alpha     =   0.1   #How important are the customers?

m.setObjective(M * gp.quicksum((tau[i ]- e[i]) for i in  P)\
               + ( 1-alpha) * gp.quicksum( distances[i,j] * x[i,j,k] for i in P for j in P for k in K )\
               +  alpha*gp.quicksum( distances[i,j] * x[i,j,k] for i in D for j in D for k in K ) ,GRB.MINIMIZE)
# m.setObjective(z.prod(priority) + gp.quicksum( 0.01 * M  * (xa[j] + xb[j]) for j in C),GRB.MINIMIZE)

m.optimize()


status = m.Status





# # Create model (the name does not matter):
# model = Model('VRP')


# # Add decision variables -- one per link
# x = model.addVars(links, vtype=GRB.BINARY, name ="link")

# # # Create variables
# #     x = model.addVar(vtype=GRB.BINARY, name="x")
# #     y = model.addVar(vtype=GRB.BINARY, name="y")
# #     z = model.addVar(vtype=GRB.BINARY, name="z")

# #     # Set objective
# #     model.setObjective(x + y + 2 * z, GRB.MAXIMIZE)


# # Constaints:
# origin      = 28  #start at location 28
# destination = 25  #end at location 25

# for i in range(1, N+1):
#     model.addConstr( sum(x[i,j] for i,j in links.select(i, '*')) - sum(x[j,i] for j,i in links.select('*',i)) ==
#                      (1 if i==origin else -1 if i==destination else 0 ),'node%s_' % i )


# print("Optimising....")
# model.optimize()

# # Print results:
# path = gp.tuplelist()

# if model.status == GRB.Status.OPTIMAL:
#    print('\n \n The final path is:')
#    for i,j in links:
#        if(x[i,j].x > 0):
#            path.append((i, j))


#    # Reorder the path:
#    que = len(path)
#    previous = origin

#    while que:

#        for i in range(len(path)):
#            if(path[i][0] == previous):
#                print(path[i])
#                previous = path[i][1]
#                que-=1








