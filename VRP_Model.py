
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
from gurobipy import Model,GRB,LinExpr
from copy import deepcopy
import gurobipy as gp
from itertools import chain, combinations
from statistics import mean
from datetime import datetime, date

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


complex = True

P,C,D,e,c,q,coord_airbase,coord_clients,coord_pizzerias,distances = getData(complex)


# Buy some drones:
K                     = range(4)                 # number of drones
droneSpeed            = 10
droneCapacity         = 14
droneEnduranceMinutes = 30
droneRange            = 1000
drone                 = UAV(droneSpeed,droneCapacity,droneEnduranceMinutes*60,droneRange)    # drone model
delay                 = 30                       # delay in seconds between drone launch / land


#+++++++++++++++++++++++++++++++++++ Set-up logger +++++++++++++++++++++++++++++++++++++++++++++++
now = datetime.now()
timeString = now.strftime("%d-%m-%Y_%H-%M-%S")

logName = 'D'+str(droneSpeed)+'-'+str(droneCapacity)+'-'+str(droneEnduranceMinutes)+'_d'+str(delay)+'_P'+str(len(P))+'_C'+str(len(C))+'_'+timeString+'.txt'
print("Log created as " + str(logName))

#++++++++++++++++++++++++++++++++ Create model+++++++++++++++++++++++++++++++++++++++++++++++++++++
m = gp.Model("VRP")

gapParameter = 0.05
methodParameter = -1

if complex:
    m.setParam('MIPGap',gapParameter)

m.setParam('Method', methodParameter)
m.setParam("LogFile", 'logs/'+logName)

#+++++++++++++++++++++++++++++++ Decision variables +++++++++++++++++++++++++++++++++++++++++++++++++

# Edge assignment to drone:
x = {}
# Add edges from airbase to pizzerias
for j in P:
    for k in K:
        x[0, j, k] = m.addVar(lb=0, ub=1, vtype=GRB.BINARY,name="x[%s,%s,%s]"%(0, j, k))
# Add edges from pizzeria to pizzerias
# for i in P:
#     for j in P:
#         for k in K:
#             if i!=j:
#                 x[i, j, k] = m.addVar(lb=0, ub=1, vtype=GRB.BINARY,name="x[%s,%s,%s]"%(i, j, k))


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
        tau[i,k] = m.addVar(lb=0, ub = max(c[:, 1]), vtype = GRB.CONTINUOUS, name = 'tau[%s,%s]'%(i, k))
    for j in C:
        tau[j,k] = m.addVar(lb=0, ub = max(c[:, 1]), vtype = GRB.CONTINUOUS, name = 'tau[%s,%s]'%(j, k))

# Launch times for each drone
launch = {}
for k in K:
    launch[k] = m.addVar(lb=0, ub = max(c[:, 0]), vtype = GRB.CONTINUOUS, name = 'launch[%s]'%(k))

# Land time for each drone
land = {}
for k in K:
    land[k] = m.addVar(lb=0, vtype = GRB.CONTINUOUS, name = 'land[%s]'%(k))

# Help binary variable for either-or constraint of launch times (y) and landing times (q)
y_1 = {}
y_2 = {}
for combi in combinations(K, 2):
    y_1[combi] = m.addVar(vtype = GRB.BINARY, name = 'y_1[%s,%s]'%(combi[0], combi[1]))
    y_2[combi] = m.addVar(vtype = GRB.BINARY, name = 'y_2[%s,%s]'%(combi[0], combi[1]))

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
m.addConstrs(( launch[k] + delay*x_k[k]*x_k[m] <= launch[m] + M * y_1[k,m] for k,m in combinations(K, 2)), name = "either launch time of drone m is D time after launch time of drone k")
m.addConstrs(( launch[m] + delay*x_k[k]*x_k[m] <= launch[k] + M * (1 - y_1[k,m]) for k,m in combinations(K, 2)), name = "or launch time of drone k is D time after launch time of drone m")

# 12 Either - or delay constraint for land time
m.addConstrs(( land[k] + delay*x_k[k]*x_k[m] <= land[m] + M * y_2[k,m] for k,m in combinations(K, 2)), name = "either land time of drone m is D time after land time of drone k")
m.addConstrs(( land[m] + delay*x_k[k]*x_k[m] <= land[k] + M * (1 - y_2[k,m]) for k,m in combinations(K, 2)), name = "or land time of drone k is D time after land time of drone m")

# 13 Max endurance
m.addConstrs(( land[k] - launch[k] <= drone.E for k in K), name = "max endurance of drone")

# 14 Capacity constriants
m.addConstrs((gp.quicksum(q[j] * x[i,j,k] for i in C for j in C if i!=j) + (gp.quicksum(q[j] * x[i,j,k] for j in C for i in P) ) <= drone.q for k in K), name = "Capacity")


m.update()



# ++++++++++++++++++++++++++++    Objective +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Distance
obj1 = LinExpr()
for key in x:
    obj1 += x[key]*distances[key[0], key[1]] #part of objective function related to total distance

m.setObjectiveN(obj1, 0, 2)

# #Total time in air
# obj3 = LinExpr()
# for k in K:
#     obj3 += (land[k]-launch[k]) # minimise time in air, dont stay in air if unnecessary

# m.setObjectiveN(obj3, 1, 1)

# Delay at locations
obj4 = LinExpr()
for i in P:
    for k in K:
        obj4 += (tau[i,k]) # minimise time delay vs expected arrival time at pizzeria
for j in C:
    for k in K:
        obj4 += (tau[j,k]) # minimise time delay vs expected arrival time at pizzeria

m.setObjectiveN(obj4, 2, 0)




m.ModelSense = GRB.MINIMIZE


m.update()


#+++++++++++++++++++++++++++++++++++ Solve Model +++++++++++++++++++++++++++++++++++++++++++++++++++

# m.write("VRP_basic.lp")


# m.computeIIS()

m.optimize()


status = m.Status


#+++++++++++++++++++++++++++++++++++ Add extra info  to log +++++++++++++++++++++++++++++++++++++++++++++++++++

file_object = open('logs/'+logName, 'a')

file_object.write('Drone speed: ' + str(droneSpeed)+'\n')
file_object.write('Drone capacity: ' + str(droneCapacity)+'\n')
file_object.write('Drone endurance minutes: ' + str(droneEnduranceMinutes)+'\n')
file_object.write('Drone land/launch delay seconds: ' + str(delay)+'\n\n')

file_object.write('Number of pizzarias: ' + str(len(P))+'\n')
file_object.write('Number of customers: ' + str(len(C))+'\n\n')

file_object.write('Gap parameter: ' + str(gapParameter)+'\n')
file_object.write('Method parameter: ' + str(methodParameter)+'\n\n')

file_object.write('Solution\n')
nObjectives = m.NumObj
for i in range(nObjectives):
    m.params.ObjNumber = i
    file_object.write("Objective "+str(i)+" value: "+str(m.ObjNVal)+'\n')

for var in m.getVars():
    if var.x:
        file_object.write('%s %f \n' % (var.varName,var.x))

# print(m.getAttr('Slack',m.getConstrs()))
# sensitivity doesnt work
# file_object.write('\n')
# file_object.write('Sensitivity analysis\n\n')
#
# file_object.write('Name\tFinal Value\tReduced Cost\tAllowable Coeff Increase\tAllowable Coeff Decrease\tLower Bound\tUpper Bound')
# for var in m.getVars():
#     print(model.getAttr("Pi", model.getConstrs()))
#     # file_object.write('\t'.join([var.VarName, str(var.x), str(var.LB), str(var.UB)]))
#
# file_object.write('\n')
#
# file_object.write('Name\tShadow Price\tRHS\tSlack\tLower Range\tUpper Range')
# for constr in m.getConstrs():
#     file_object.write('\t'.join([constr.ConstrName, constr.pi, constr.RHS, constr.Slack, constr.SARHSLow, constr.SARHSUp]))



print("Wrote file: " + str(logName))
file_object.close()
