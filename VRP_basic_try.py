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



class UAV:
    """
    The class for our drones.

    (We can add multiple modles of drones, I don't know how that will look like in the model...')

    """
    def __init__(self, maxspeed, capacity, endurance, maxrange):

        self.v = maxspeed # [m/s]
        self.q = capacity  # [pizzaz]   1 pizza = 1kg
        self.R = maxrange  # [m]
        self.E = endurance  #[s]


# Load data:
data    = np.genfromtxt("dummy_data.csv",skip_header=1,delimiter=',',dtype=int)

# Data format: node1, node2, distance


# Nodes (except deport) & Links:
N       = int(max(max(data[:,0]),max(data[:,1])))
L       = int(len(data[:,0]))

# Customers:
C       = range(N)

# Distances:
D       = data[:,2]

# Buy some drones:
K       = range(2)                      # number of drones
drone   = UAV(1,1,30*60,1000)    # drone model



# Build the graph as a list of tuples:
links = gp.tuplelist()
cost  = {}

for i in range(L):

    from_node = data[i,0]
    to_node   = data[i,1]
    cost_arc  = data[i,2]

    links.append((from_node,to_node))

    cost[from_node, to_node] = cost_arc

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
x = m.addVars(L, L, K, vtype=GRB.BINARY, name="x")

# # Technician cannot leave or return to a depot that is not its base
# for k in technicians:
#     for d in D:
#         if k.depot != d:
#             for i in L:
#                 y[i,d,k.name].ub = 0
#                 y[d,i,k.name].ub = 0

# Start time of service
t = m.addVars(L, ub=600, name="t")

# Lateness of service
z = m.addVars(C, name="z")

# Artificial variables to correct time window upper and lower limits
xa = m.addVars(C, name="xa")
xb = m.addVars(C, name="xb")


#+++++++++++++++++++++++++++++++++ Constraints  ++++++++++++++++++++++++++++++++++++++++++++++++++

# Each customer needes to be visited:
m.addConstrs((gp.quicksum(x[i,j,k]  for k in K for j in C)== 1 for i in C), name="visited cutomer")

# Each drone that deleivered must leave:
m.addConstrs((gp.quicksum(x[j,i,k] for j in C) == gp.quicksum(x[i,j,k] for j in C) for i in C for k in K), name="leave customer")

# Technician capacity constraints (3)
capLHS = {k : gp.quicksum(dur[j]*x[j,k] for j in C) +\
    gp.quicksum(dist[i,j]*y[i,j,k] for i in L for j in L) for k in K}

m.addConstrs((capLHS[k] <= drone.q), name="Capacity")

# Technician tour constraints (4 and 5)
m.addConstrs((y.sum('*', loc[j], k) == x[j,k] for k in K for j in C),\
    name="techTour1")
m.addConstrs((y.sum(loc[j], '*', k) == x[j,k] for k in K for j in C),\
    name="techTour2")

# Same depot constraints (6 and 7)
m.addConstrs((gp.quicksum(y[j,depot[k],k] for j in J) == u[k] for k in K),\
    name="sameDepot1")
m.addConstrs((gp.quicksum(y[depot[k],j,k] for j in J) == u[k] for k in K),\
    name="sameDepot2")

# Temporal constraints (8) for customer locations
M = {(i,j) : 600 + dur[i] + dist[loc[i], loc[j]] for i in C for j in C}
m.addConstrs((t[loc[j]] >= t[loc[i]] + dur[i] + dist[loc[i], loc[j]]\
    - M[i,j]*(1 - gp.quicksum(y[loc[i],loc[j],k] for k in K))\
    for i in C for j in C), name="tempoCustomer")

# Temporal constraints (8) for depot locations
M = {(i,j) : 600 + dist[i, loc[j]] for i in D for j in C}
m.addConstrs((t[loc[j]] >= t[i] + dist[i, loc[j]]\
    - M[i,j]*(1 - y.sum(i,loc[j],'*')) for i in D for j in C),\
    name="tempoDepot")

# Time window constraints (9 and 10)
m.addConstrs((t[loc[j]] + xa[j] >= tStart[j] for j in C), name="timeWinA")
m.addConstrs((t[loc[j]] - xb[j] <= tEnd[j] for j in C), name="timeWinB")

# Lateness constraint (11)
m.addConstrs((z[j] >= t[loc[j]] + dur[j] - tDue[j] for j in C),\
    name="lateness")

### Objective function
M = 6100

m.setObjective(z.prod(priority) + gp.quicksum( 0.01 * M  * (xa[j] + xb[j]) for j in C),
               GRB.MINIMIZE)


m.optimize()

status = m.Status
if status in [GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED]:
    print("Model is either infeasible or unbounded.")
    sys.exit(0)
elif status != GRB.OPTIMAL:
    print("Optimization terminated with status {}".format(status))
    sys.exit(0)

### Print results
# Assignments
print("")
for j in customers:
    if g[j.name].X > 0.5:
        jobStr = "Nobody assigned to {} ({}) in {}".format(j.name,j.job.name,j.loc)
    else:
        for k in K:
            if x[j.name,k].X > 0.5:
                jobStr = "{} assigned to {} ({}) in {}. Start at t={:.2f}.".format(k,j.name,j.job.name,j.loc,t[j.loc].X)
                if z[j.name].X > 1e-6:
                    jobStr += " {:.2f} minutes late.".format(z[j.name].X)
                if xa[j.name].X > 1e-6:
                    jobStr += " Start time corrected by {:.2f} minutes.".format(xa[j.name].X)
                if xb[j.name].X > 1e-6:
                    jobStr += " End time corrected by {:.2f} minutes.".format(xb[j.name].X)
    print(jobStr)

# Technicians
print("")
for k in technicians:
    if u[k.name].X > 0.5:
        cur = k.depot
        route = k.depot
        while True:
            for j in customers:
                if y[cur,j.loc,k.name].X > 0.5:
                    route += " -> {} (dist={}, t={:.2f}, proc={})".format(j.loc, dist[cur,j.loc], t[j.loc].X, j.job.duration)
                    cur = j.loc
            for i in D:
                if y[cur,i,k.name].X > 0.5:
                    route += " -> {} (dist={})".format(i, dist[cur,i])
                    cur = i
                    break
            if cur == k.depot:
                break
        print("{}'s route: {}".format(k.name, route))
    else:
        print("{} is not used".format(k.name))


# Utilization
print("")
for k in K:
    used = capLHS[k].getValue()
    total = cap[k]
    util = used / cap[k] if cap[k] > 0 else 0
    print("{}'s utilization is {:.2%} ({:.2f}/{:.2f})".format(k, util,\
        used, cap[k]))
totUsed = sum(capLHS[k].getValue() for k in K)
totCap = sum(cap[k] for k in K)
totUtil = totUsed / totCap if totCap > 0 else 0
print("Total technician utilization is {:.2%} ({:.2f}/{:.2f})".format(totUtil, totUsed, totCap))







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








