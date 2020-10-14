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

###################
### MODEL SETUP ###
###################

# Keep track of start time to compute overall comptuational performance
startTimeSetUp = time.time()
# Initialize empty model
model = Model()


# Model parameters
letters = ['A','B','C','D','E','F','G','H','I','J','FINAL']
# Normal length of each ativity
ub_normal_length = [32,28,36,16,32,54,17,20,34,18,100]
# Crash lenth
ub_crash         = [4,3,5,3,5,7,2,3,4,2]
# Crash cost (per week of reduction)
cost_crash       = [5,7,8,4,5,6,3,4,9,2]
# Targeted completion time
max_time         = 92

#################
### VARIABLES ###
#################

# Variables are defined as dictionaries, using the same indices you would
# use in your MILP formulation


# Here I am using numbers for indices insteaf of letrers, using the following sequence:
# y_0 = y_A, y_1 = y_B etc
# Also, the upper bound of each y variable could be lowered keeping into
# consideration precedence constraints (for example, if we want tp compelte the project
# in 92 weeks, the upper bound on y_A should be lower than 92 weeks which
# is the current value). Here it does not really affect performances, but in
# larger problems you should be more careful
y = {}
for i in range(0,len(letters)):
    y[i]=model.addVar(lb=0, ub=max_time, vtype=GRB.INTEGER,name="y[%s]"%(letters[i]))

# Same idea for x variables. Here I am using the proper upper bound    
x = {}
for i in range(0,len(letters)-1):
    x[i]=model.addVar(lb=0, ub=ub_crash[i], vtype=GRB.INTEGER,name="x[%s]"%(letters[i]))

            
# Update the model (important!) so all variables are added to the model  
model.update()

###################
### CONSTRAINTS ###
###################

# Similar to variable definition. Note: there are multiple ways to define 
# constraints. My approach is the following
# 1) for each constraint, I define a LinExpr() object where I will save the 
# left hand side of my constraint
# 2) I "populate" each left hand side adding the needed decision variables
# multiplied by their coefficient for that constraint
# 3) I define the "sense" of the constraint (<=, =, >=) and the right hand
# side. Since I bring all decision variables to the left hand side, the right hand
# is just a number

# Note: here I manually specified all constraints because there were not too many.
# In larger problems, please define a file with your constraint set and
# automte this process

thisLHS = LinExpr()
thisLHS += y[2]-y[0]+x[0]
model.addConstr(lhs=thisLHS, sense=GRB.GREATER_EQUAL, rhs=ub_normal_length[0],
                         name='A_C')

thisLHS = LinExpr()
thisLHS += y[9]-y[2]+x[2]
model.addConstr(lhs=thisLHS, sense=GRB.GREATER_EQUAL, rhs=ub_normal_length[2],
                         name='C_J')

thisLHS = LinExpr()
thisLHS += y[5]-y[1]+x[1]
model.addConstr(lhs=thisLHS, sense=GRB.GREATER_EQUAL, rhs=ub_normal_length[1],
                         name='B_F')

thisLHS = LinExpr()
thisLHS += y[9]-y[5]+x[5]
model.addConstr(lhs=thisLHS, sense=GRB.GREATER_EQUAL, rhs=ub_normal_length[5],
                         name='F_J')

thisLHS = LinExpr()
thisLHS += y[4]-y[1]+x[1]
model.addConstr(lhs=thisLHS, sense=GRB.GREATER_EQUAL, rhs=ub_normal_length[1],
                         name='B_E')

thisLHS = LinExpr()
thisLHS += y[7]-y[4]+x[4]
model.addConstr(lhs=thisLHS, sense=GRB.GREATER_EQUAL, rhs=ub_normal_length[4],
                         name='E_H')

thisLHS = LinExpr()
thisLHS += y[3]-y[1]+x[1]
model.addConstr(lhs=thisLHS, sense=GRB.GREATER_EQUAL, rhs=ub_normal_length[1],
                         name='B_D')

thisLHS = LinExpr()
thisLHS += y[6]-y[3]+x[3]
model.addConstr(lhs=thisLHS, sense=GRB.GREATER_EQUAL, rhs=ub_normal_length[3],
                         name='D_G')

thisLHS = LinExpr()
thisLHS += y[8]-y[6]+x[6]
model.addConstr(lhs=thisLHS, sense=GRB.GREATER_EQUAL, rhs=ub_normal_length[6],
                         name='G_I')

thisLHS = LinExpr()
thisLHS += y[8]-y[4]+x[4]
model.addConstr(lhs=thisLHS, sense=GRB.GREATER_EQUAL, rhs=ub_normal_length[4],
                         name='E_I')

thisLHS = LinExpr()
thisLHS += y[7]-y[6]+x[6]
model.addConstr(lhs=thisLHS, sense=GRB.GREATER_EQUAL, rhs=ub_normal_length[6],
                         name='G_H')

thisLHS = LinExpr()
thisLHS += y[10]-y[9]+x[9]
model.addConstr(lhs=thisLHS, sense=GRB.GREATER_EQUAL, rhs=ub_normal_length[9],
                         name='J_FIN')

thisLHS = LinExpr()
thisLHS += y[10]-y[7]+x[7]
model.addConstr(lhs=thisLHS, sense=GRB.GREATER_EQUAL, rhs=ub_normal_length[7],
                         name='H_FIN')

thisLHS = LinExpr()
thisLHS += y[10]-y[8]+x[8]
model.addConstr(lhs=thisLHS, sense=GRB.GREATER_EQUAL, rhs=ub_normal_length[8],
                         name='I_FIN')

thisLHS = LinExpr()
thisLHS += y[10]
model.addConstr(lhs=thisLHS, sense=GRB.LESS_EQUAL, rhs=max_time,
                         name='FIN_MAX')


# Same as above. This is necessary to have all the constraints in your model
model.update()
 

# Defining objective function     
obj        = LinExpr() 

# Adding each crash cost and associated decision variable to the
# objective function
for i in range(0,len(letters)-1):
    obj += x[i]*cost_crash[i]

# Important: here we are telling the solver we want to minimize the objective
# function. Make sure you are selecting the right option!    
model.setObjective(obj,GRB.MINIMIZE)
# Updating the model
model.update()
# Writing the .lp file. Important for debugging
model.write('model_formulation.lp')    

# Here the model is actually being optimized
model.optimize()
# Keep track of end time to compute overall comptuational performance 
endTime   = time.time()

# Saving our solution in the form [name of variable, value of variable]
solution = []
for v in model.getVars():
     solution.append([v.varName,v.x])
     
print(solution)


