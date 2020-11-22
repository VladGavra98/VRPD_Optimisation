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
from itertools import chain
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from statistics import mean
import matplotlib.image as mpimg
import matplotlib.text as mpl_text


# consts.
M = 1000000



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




# ++++++++++++++++++++++ Build data strcutures  +++++++++++++++++++++++++++++++++++++++++++++++++++

# 1.  airbase - pizzerias   (directed)   <var_name>_AP
# 2.  pizzerias - customers (directed)   <var_name>_PC
# 3.  customer - customer   (UNdirected) <var_name>_CC
# 4.  customer - airbase    (directed)   <var_name>_CA


def getData():

    # Load data:
    data_CA   = np.genfromtxt("client_airbase_distances.csv",skip_header=1,delimiter=',',dtype=int)
    data_AP   = np.genfromtxt("airbase_pizzerias_distances.csv",skip_header=1,delimiter=',',dtype=int)
    data_CC   = np.genfromtxt("client_1_client_2_distances.csv",skip_header=1,delimiter=',',dtype=int)
    data_PC   = np.genfromtxt("pizzerias_clients.csv",skip_header=1,delimiter=',',dtype=int)

    e_tab     = np.genfromtxt("pizzeria_expected_arrival_time.csv",skip_header=1,delimiter=',',dtype=int)
    # Data format: node1 lat,long, node2 lat,long , distance

    dist_AP = data_AP[:,4]
    dist_PC = data_PC[:,4]
    dist_CC = data_CC[:,4]
    dist_CA = data_CA[:,4]

    # Load data for visualisation purposes (extracting the coordinates):
    data_CA_vis = np.genfromtxt("client_airbase_distances.csv", skip_header=1, delimiter=',')
    data_AP_vis = np.genfromtxt("airbase_pizzerias_distances.csv", skip_header=1, delimiter=',')

    # saving the coordinates of the different destinations for visualisation purposes
    coord_airbase = data_AP_vis[0, 0:2]  # coordinates of the airbase
    coord_pizzerias = data_AP_vis[:, 2:4]  # coordinates of the pizzerias
    coord_clients = data_CA_vis[:, 0:2]  # coordinates of the clients

    # Get number of locations:
    Cmax = len(dist_CA)  # number of customers
    Pmax = len(dist_AP)  #number of pizzerias
    Dmax = Cmax + Pmax + 1  #total number of destinations


    # Orders:  time and quanitity
    e       = np.zeros( ( len(e_tab[:,2]) + 1 ) )
    e[1:]   = e_tab[:,2]      # to stay consistent,
    # inidex 0 is the airbase which has no arrival time

    q                 = np.zeros((Dmax))   # each of them wants 2 pizzas
    q[Pmax: Dmax]     = 2
    """  !!! CHANGE HERE FOR NUMBER OF PIZZAS ORDERED!!!!"""

    C = range( Pmax + 1, Dmax)# number of customers
    P = range( 1, Pmax + 1)  #number of pizzerias
    D = range( Dmax)  #total number of destinations

    if len(e) != len(P)+1:
        print(len(e),len(P))
        print("Error in the Pizzeria files!\n")

    # Assemble graph:
    # AP ,PC, CC, CA

    distances = np.zeros((Dmax,Dmax))

    for i in P:
        distances[0,i] = dist_AP[i-1]

    for i in P:
        for j in C:
            distances[i,j] = dist_PC[(i-P[0])*Cmax +j-C[0]]

    k = 0
    for i in C:
        for j in C:
            if i!= j:
                distances[i,j] = dist_CC[k]
                distances[j,i] = dist_CC[k]

                k+=1

    for i in C:
        distances[0,i] = dist_CA[i-Pmax-1]

    distances[:,0] = distances[0,:] # from i to 0


    return P,C,D,e,q,coord_airbase,coord_clients,coord_pizzerias,distances



P,C,D,e,q,coord_airbase,coord_clients,coord_pizzerias,distances = getData()

print(distances)


# Buy some drones:
K       = range(3)               # number of drones
drone   = UAV(10,10,180*60,1000)    # drone model






### Create model
m = gp.Model("VRP")

#+++++++++++++++++++++++++++++ Decision variables +++++++++++++++++++++++++++++++++++++++++++++++++

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


tau = {}
for i in P:
    tau[i] = m.addVar(vtype = GRB.INTEGER, name = 'tau')
for j in C:
    tau[j] = m.addVar(vtype = GRB.INTEGER, name = 'tau')
tau[0] = m.addVar(vtype = GRB.INTEGER, name = 'tau')

m.update()

#+++++++++++++++++++++++++++++++++ Constraints  ++++++++++++++++++++++++++++++++++++++++++++++++++

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
m.addConstrs((e[i] - tau[i] <= 0 for i in P), name ="time bound on pizzeria")

# 7 Time window for arriving at pizzeria:
m.addConstrs(((0 + distances[0,j]/drone.v - (1 - x[0,j,k]) * M)  <= tau[j] for j in P for k in K), name = "Time window pizzeria")   # the first 0 might change later if drones leave at differetn times

m.addConstrs(((tau[i] + distances[i,j]/drone.v - (1 - x[i,j,k]) * M) <= tau[j] for i in chain(P, C) for j in C for k in K if i!=j), name = "Time window pizzeria to customer and customer to customer")
# tau[j] has no upper bound except for endurance

# # 8 Endurance constraints
m.addConstrs((tau[i] <= drone.E - distances[i,0]/drone.v  for i in C), name = "Endurance1") # we need to change endurance constraint
m.addConstr((tau[0] <= drone.E), name = "Endurance2")

m.update()
# ++++++++++++++++++++++++++++ Objective +++++++++++++++++++++++++++++++++++++++++++++++++++++++++



alpha     =   1   #How important are the customers?
obj = LinExpr()

for key in x:
    obj += (1-alpha)*x[key]*distances[key[0], key[1]] #part of objective function related to total distance
    if key[0] in P or key[0] in C:
        #part of objective function related to distance from pizzeria to customer and customer to customer
        obj += alpha*x[key]*distances[key[0], key[1]]
for i in P:
    obj += M*(tau[i]-e[i]) # minimise time delay vs expected arrival time at pizzeria

#for j in chain(C, P):
#    obj += M*tau[j] # minimise arrival times( otherwise tau goes to upper bound) maybe try to avoid implementing this



m.setObjective(obj, GRB.MINIMIZE)


m.update()


m.write("VRP_basic.lp")



m.optimize()


status = m.Status

# ++++++++++++++++++++++++++++++ Printing & Visualisation ++++++++++++++++++++++++++++++++++++++++++

for var in m.getVars():
    if var.x:
        print('%s %f' % (var.varName,var.x))


def visualisation(print_tau):

    imData = plt.imread("map_first_try_basic_model.JPG") #first we are plotting the background image

    fig, ax = plt.subplots()
    ax.imshow(imData, extent=[4.3458, 4.3954, 51.98554, 52.02264]) #setting the corners of our plot; these points work for well for the initial dataset

    colours=["r","b","c"] #the route of the first drone will be shown in red, of the second one in blue and of the third one in cyan

    for var in m.getVars():
        if var.x and var.varName[0]=="x" and var.varName[1]=="[": #for plotting, we are interested in the x[i,j,k] variables

            if var.varName[2]=="0": #scenario 1: we are at the airbase, going to the pizzerias
                y_coord=[coord_airbase[0],coord_pizzerias[int(var.varName[4])-1][0]] #the y_coord is the lattitude (North)
                x_coord=[coord_airbase[1],coord_pizzerias[int(var.varName[4])-1][1]] #the x_coord is the longitude (East)
                ax.plot(x_coord, y_coord, colours[int(var.varName[6])], linewidth=2.5)

                # ---- plotting the drone icons ----
                arr_drone = mpimg.imread("Drone_white_icon.png")
                imagebox = OffsetImage(arr_drone, zoom=0.04)
                ab = AnnotationBbox(imagebox, (mean(x_coord), mean(y_coord)), frameon=False)
                ax.add_artist(ab)
                ax.add_artist(mpl_text.Text(x=mean(x_coord), y=mean(y_coord), text=str(var.varName[6]), weight="bold", color='black', fontsize=9, verticalalignment='center',horizontalalignment='center'))


            if int(var.varName[2])>0 and int(var.varName[2])<=len(P): #scenario 2: we are at a pizzeria, going to a customer
                y_coord=[coord_pizzerias[int(var.varName[2])-1][0], coord_clients[int(var.varName[4])-len(P)-1][0]]
                x_coord=[coord_pizzerias[int(var.varName[2])-1][1], coord_clients[int(var.varName[4])-len(P)-1][1]]
                ax.plot(x_coord, y_coord, colours[int(var.varName[6])], linewidth=2.5)

            if int(var.varName[2])>len(P) and int(var.varName[4])>len(P): #scenario 3: we are at a customer, going to another customer
                y_coord = [coord_clients[int(var.varName[2])-len(P)-1][0], coord_clients[int(var.varName[4])-len(P)- 1][0]]
                x_coord = [coord_clients[int(var.varName[2])-len(P)-1][1], coord_clients[int(var.varName[4])-len(P)- 1][1]]
                ax.plot(x_coord, y_coord, colours[int(var.varName[6])], linewidth=2.5)

            if int(var.varName[2])>len(P) and int(var.varName[4])==0: #scenario 4: we are at a customer, going back to the airbase
                y_coord = [coord_clients[int(var.varName[2]) - len(P) - 1][0],coord_airbase[0]]
                x_coord = [coord_clients[int(var.varName[2]) - len(P) - 1][1],coord_airbase[1]]
                ax.plot(x_coord, y_coord, colours[int(var.varName[6])], linewidth=2.5)

    # -----plotting the info about the airbase------
    ax.plot((coord_airbase[1]), (coord_airbase[0]), 'w*', markersize=12) #airbase as white star

    ax.text((coord_airbase[1]), (coord_airbase[0])-0.001, 'Airbase', color='white', fontsize=10, bbox={'facecolor': 'red', 'alpha': 0.6, 'pad': 2})
    if (print_tau == True):
        ax.text((coord_airbase[1]), (coord_airbase[0])-0.002, r'$\tau$' + "=" + str(tau[0].x), color='white', fontsize=8, bbox={'facecolor': 'red', 'alpha': 0.6, 'pad': 2})


    # -----plotting the info about the pizzerias------
    ax.plot((coord_pizzerias[:, 1]), (coord_pizzerias[:, 0]), 'w^', markersize=7) #pizzerias white triangles

    for i in range(len(coord_pizzerias)):
        ax.text((coord_pizzerias[i,1]), (coord_pizzerias[i,0]) - 0.001, str(i+1), color='white', fontsize=10,
                bbox={'facecolor': 'red', 'alpha': 0.6, 'pad': 2})
        if (print_tau == True):
            ax.text((coord_pizzerias[i,1]), (coord_pizzerias[i,0])-0.002, r'$\tau$' + "=" + str(tau[i+1].x), color='white', fontsize=8, bbox={'facecolor': 'red', 'alpha': 0.6, 'pad': 2})


    # -----plotting the info about the clients------
    ax.plot((coord_clients[:,1]), (coord_clients[:,0]), 'wo') #clients as white dots

    for i in range(len(coord_clients)):
        ax.text((coord_clients[i,1]), (coord_clients[i,0]) - 0.001, str(i+1+len(P)), color='white', fontsize=10,
                bbox={'facecolor': 'red', 'alpha': 0.6, 'pad': 2})
        if (print_tau == True):
            ax.text((coord_clients[i,1]), (coord_clients[i,0])-0.002, r'$\tau$' + "=" + str(tau[i+1+len(P)].x), color='white', fontsize=8, bbox={'facecolor': 'red', 'alpha': 0.6, 'pad': 2})

    plt.xlabel("Longitutde (" + u"\N{DEGREE SIGN}" + "E)")
    plt.ylabel("Latitude (" + u"\N{DEGREE SIGN}" + "N)")
    plt.show()

#Comment/uncomment the following line in order to hide/see the visualisation of the current solution
visualisation(True)    #write True if you want to also plot the taus. Write False if you don't want the taus to be plotted

def verify_cross_over(verify):
    if verify==True:
        distance_cross_over = distances[0, 1] + distances[1, 6] + distances[6, 5] + distances[5, 4] + distances[4, 0]
        distance_no_cross_over = distances[0, 1] + distances[1, 5] + distances[5, 4] + distances[4, 6] + distances[6, 0]

        print(distance_cross_over, distance_no_cross_over)

        if distance_cross_over > distance_no_cross_over:
            print("Cross over implies larger distance")
        else:
            print("Not crossing over implies larger distance")

        print(distances)

verify_cross_over(True)




