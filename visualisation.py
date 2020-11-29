# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 18:54:25 2020

@author: vladg
"""

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from statistics import mean
import matplotlib.image as mpimg
import matplotlib.text as mpl_text

from VRP_Model import *

largeFont = True
# ++++++++++++++++++++++++++++++ Printing & Visualisation ++++++++++++++++++++++++++++++++++++++++++
# totalDistance = 0
# for var in m.getVars():
#     if round(var.x) != 0 and var.varName[0] == "x" and var.varName[1] == "[":
#         name_of_variable = var.varName[2:-1].split(",")
#         totalDistance += distances[int(name_of_variable[0]), int(name_of_variable[1])]
# print("Total distance is: ", totalDistance)

nObjectives = m.NumObj
for i in range(nObjectives):
    m.params.ObjNumber = i
    print("Objective "+str(i)+" value: "+str(m.ObjNVal))

for var in m.getVars():
    if var.x:
        print('%s %f' % (var.varName,var.x))



def visualisation(print_tau):

    imData = plt.imread("map_first_try_basic_model.JPG") #first we are plotting the background image

    fig, ax = plt.subplots(1,1,figsize=(10,7.5)) #w, h

    ax.set_title("Objective (total distance): " + str(round(m.objVal)) + " m", fontsize = 14)
    ax.imshow(imData, extent=[4.3458, 4.3954, 51.98554, 52.02264]) #setting the corners of our plot; these points work for well for the initial dataset

    #colours= ["#540d6e","#ee4266","#ffd23f","#3bceac","#0ead69","#f94144","#f3722c","#f8961e","#f9844a","#f9c74f","#90be6d","#8ad0bb","#4d908e","#8da6b9","#277da1"] #the route of the first drone will be shown in red, of the second one in blue and of the third one in cyan
    colours = ["b", "yellow", "r", "magenta", "darkorange", "springgreen", "plum", "aqua", "lightpink", "k"]
    for var in m.getVars():
        if round(var.x) != 0 and var.varName[0]=="x" and var.varName[1]=="[": #for plotting, we are interested in the x[i,j,k] variables
            name_of_variable_1 = var.varName[2:-1].split(",")
            if name_of_variable_1[0]=="0": #scenario 1: we are at the airbase, going to the pizzerias
                y_coord=[coord_airbase[0],coord_pizzerias[int(name_of_variable_1[1])-1][0]] #the y_coord is the lattitude (North)
                x_coord=[coord_airbase[1],coord_pizzerias[int(name_of_variable_1[1])-1][1]] #the x_coord is the longitude (East)
                ax.plot(x_coord, y_coord, colours[int(name_of_variable_1[2])], linewidth=2.5)

                # ---- plotting the drone icons ----
                arr_drone = mpimg.imread("Drone_white_icon.png")
                imagebox = OffsetImage(arr_drone, zoom=0.04)
                ab = AnnotationBbox(imagebox, (mean(x_coord), mean(y_coord)), frameon=False)
                ax.add_artist(ab)
                ax.add_artist(mpl_text.Text(x=mean(x_coord), y=mean(y_coord), text=str(name_of_variable_1[2]), weight="bold", color='black', fontsize=9, verticalalignment='center',horizontalalignment='center'))


            if int(name_of_variable_1[0])>0 and int(name_of_variable_1[0])<=len(P): #scenario 2: we are at a pizzeria, going to a customer
                y_coord=[coord_pizzerias[int(name_of_variable_1[0])-1][0], coord_clients[int(name_of_variable_1[1])-len(P)-1][0]]
                x_coord=[coord_pizzerias[int(name_of_variable_1[0])-1][1], coord_clients[int(name_of_variable_1[1])-len(P)-1][1]]
                ax.plot(x_coord, y_coord, colours[int(name_of_variable_1[2])], linewidth=2.5)

            if int(name_of_variable_1[0])>len(P) and int(name_of_variable_1[1])>len(P): #scenario 3: we are at a customer, going to another customer
                y_coord = [coord_clients[int(name_of_variable_1[0])-len(P)-1][0], coord_clients[int(name_of_variable_1[1])-len(P)- 1][0]]
                x_coord = [coord_clients[int(name_of_variable_1[0])-len(P)-1][1], coord_clients[int(name_of_variable_1[1])-len(P)- 1][1]]
                ax.plot(x_coord, y_coord, colours[int(name_of_variable_1[2])], linewidth=2.5)

            if int(name_of_variable_1[0])>len(P) and int(name_of_variable_1[1])==0: #scenario 4: we are at a customer, going back to the airbase
                y_coord = [coord_clients[int(name_of_variable_1[0]) - len(P) - 1][0],coord_airbase[0]]
                x_coord = [coord_clients[int(name_of_variable_1[0]) - len(P) - 1][1],coord_airbase[1]]
                ax.plot(x_coord, y_coord, colours[int(name_of_variable_1[2])], linewidth=2.5)


    # -----plotting the info about the airbase------
    ax.plot((coord_airbase[1]), (coord_airbase[0]), 'w*', markersize=12) #airbase as white star

    ax.text((coord_airbase[1]), (coord_airbase[0])-0.001, 'Airbase', color='white', fontsize = 12 if largeFont else 10, bbox={'facecolor': 'red', 'alpha': 0.6, 'pad': 2})


    # -----plotting the info about the pizzerias------
    ax.plot((coord_pizzerias[:, 1]), (coord_pizzerias[:, 0]), 'w^', markersize=7) #pizzerias white triangles

    for i in range(len(coord_pizzerias)):
        for j in K:
                if round(tau[i+1,j].x) != 0:
                    ax.text((coord_pizzerias[i,1]), (coord_pizzerias[i,0])-0.001, str(i + 1) +  ", " + r'$\tau$' + "=" + str(int(tau[i+1,j].x)), color='white',  fontsize = 12 if largeFont else 8, bbox={'facecolor': 'red', 'alpha': 0.6, 'pad': 1})



    # -----plotting the info about the clients------
    ax.plot((coord_clients[:,1]), (coord_clients[:,0]), 'wo') #clients as white dots

    for i in range(len(coord_clients)):
        if (print_tau == True):
            for j in K:
                if round(tau[i+1+len(P),j].x) != 0:
                    ax.text((coord_clients[i,1]), (coord_clients[i,0])-0.001, str(i+1+len(P)) + ", " + r'$\tau$' + "=" + str(int(tau[i+1+len(P),j].x)), color='white', fontsize = 12 if largeFont else 8, bbox={'facecolor': 'red', 'alpha': 0.6, 'pad': 1})

        else:
            ax.text((coord_clients[i, 1]), (coord_clients[i, 0]) - 0.001, str(i + 1 + len(P)), color='white',
                    fontsize=10,
                    bbox={'facecolor': 'red', 'alpha': 0.6, 'pad': 2})

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    plt.xlabel("Longitude (" + u"\N{DEGREE SIGN}" + "E)", fontsize = 14)
    plt.ylabel("Latitude (" + u"\N{DEGREE SIGN}" + "N)", fontsize = 14)

    plt.tight_layout()
    plt.show()



#Comment/uncomment the following line in order to hide/see the visualisation of the current solution
visualisation(True)    #write True if you want to also plot the taus. Write False if you don't want the taus to be plotted


# ++++++++++++++++++++++++++++++ Verifying cross-over ++++++++++++++++++++++++++++++++++++++++++

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

verify_cross_over(False)