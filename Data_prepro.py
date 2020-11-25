# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 18:49:59 2020

@author: vladg,danny,flori

"""

import numpy as np
import time
import os
from itertools import chain, combinations


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

    # Timing data:
    e_tab     = np.genfromtxt("pizzeria_expected_arrival_time.csv",skip_header=1, delimiter=',', dtype=int)
    c_tab     = np.genfromtxt("customer_arrival_time.csv", skip_header=1, delimiter=',', dtype=int)

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

    # Time at customers:
    c         =  np.zeros( ( c_tab.shape[0] + C[0], c_tab.shape[1] ) )
    c[C[0]:]  =  c_tab

    if len(e) != len(P)+1:
        print(len(e),len(P))
        print("Error in the Pizzeria files!\n")

    if c.shape[0] != len(C) + len(P) +1:
        print(c.shape[0],len(C))
        print("Error in the Customer files!\n")

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

    return P,C,D,e,c,q,coord_airbase,coord_clients,coord_pizzerias,distances
