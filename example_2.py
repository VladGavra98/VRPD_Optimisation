# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 22:22:35 2020

@author: abombelli
"""

# Loading packages that are used in the code
import numpy as np
import os
import pandas as pd
import time
from gurobipy import Model,GRB,LinExpr
import pickle
from copy import deepcopy

# Get path to current folder
cwd = os.getcwd()

# Get all instances
full_list           = os.listdir(cwd)

# instance name
instance_name = 'data_example_2.xlsx'

# Load data for this instance
edges                 = pd.read_excel(os.path.join(cwd,instance_name),sheet_name='Airport data')

startTimeSetUp = time.time()
model = Model()

#################
### VARIABLES ###
#################
x = {}
for i in range(0,len(edges)):
    x[edges['From'][i],edges['To'][i]]=model.addVar(lb=0, ub=1, vtype=GRB.BINARY,name="x[%s,%s]"%(edges['From'][i],edges['To'][i]))

            
model.update()

###################
### CONSTRAINTS ###
###################

source = 25
sink   = 28

for i in range(1,edges['From'][len(edges)-1]):
    idx_this_node_out = np.where(edges['From']==i)[0]
    idx_this_node_in  = np.where(edges['To']==i)[0]
    
    if i != source and i != sink:
        thisLHS = LinExpr()
        if len(idx_this_node_out) > 0:
            for j in range(0,len(idx_this_node_out)):
                thisLHS += x[i,edges['To'][idx_this_node_out[j]]]
        if len(idx_this_node_in) > 0:
            for j in range(0,len(idx_this_node_in)):
                thisLHS -= x[edges['From'][idx_this_node_in[j]],i]
        model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=0,
                         name='node_'+str(i))
        
    if i is source:
        thisLHS = LinExpr()
        if len(idx_this_node_out) > 0:
            for j in range(0,len(idx_this_node_out)):
                thisLHS += x[i,edges['To'][idx_this_node_out[j]]]
                model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=1,
                         name='node_'+str(i)+'_source_out')
        thisLHS = LinExpr()
        if len(idx_this_node_in) > 0:
            for j in range(0,len(idx_this_node_in)):
                thisLHS += x[edges['From'][idx_this_node_in[j]],i]
                model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=0,
                         name='node_'+str(i)+'_source_in')
                
    if i is sink:
        thisLHS = LinExpr()
        if len(idx_this_node_in) > 0:
            for j in range(0,len(idx_this_node_in)):
                thisLHS += x[edges['From'][idx_this_node_in[j]],i]
                model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=1,
                         name='node_'+str(i)+'_sink_in')
        thisLHS = LinExpr()
        if len(idx_this_node_out) > 0:
            for j in range(0,len(idx_this_node_out)):
                thisLHS += x[i,edges['To'][idx_this_node_out[j]]]
                model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=0,
                         name='node_'+str(i)+'_sink_out')
        
        
model.update()
 
       
obj        = LinExpr() 

for i in range(0,len(edges)):
    obj += edges['Distance'][i]*x[edges['From'][i],edges['To'][i]]


model.setObjective(obj,GRB.MINIMIZE)
model.update()
model.write('model_formulation.lp')    

model.optimize()
endTime   = time.time()

solution = []
     
for v in model.getVars():
     solution.append([v.varName,v.x])
     
route_complete = False
current_node   = source
path           = [source]
     
while route_complete is False:
    # Connections from current node
    idx_this_node_out = np.where(edges['From']==current_node)[0]
    #print(idx_this_node_out)
    for i in range(0,len(idx_this_node_out)):
        if x[current_node,edges['To'][idx_this_node_out[i]]].x >= 0.99:
            path.append(edges['To'][idx_this_node_out[i]])
            current_node = edges['To'][idx_this_node_out[i]]
            
            if current_node == sink:
                route_complete = True
                break
            else:
                break
            
print(path)
            
            
    
     


    
