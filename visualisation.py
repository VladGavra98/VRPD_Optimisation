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


    # -----plotting the info about the pizzerias------
    ax.plot((coord_pizzerias[:, 1]), (coord_pizzerias[:, 0]), 'w^', markersize=7) #pizzerias white triangles

    for i in range(len(coord_pizzerias)):
        ax.text((coord_pizzerias[i,1]), (coord_pizzerias[i,0]) - 0.001, str(i+1), color='white', fontsize=10,
                bbox={'facecolor': 'red', 'alpha': 0.6, 'pad': 2})
        if (print_tau == True):
            for j in K:
                if tau[i+1,j].x:
                    ax.text((coord_pizzerias[i,1]), (coord_pizzerias[i,0])-0.002, r'$\tau$' + "=" + str(round(tau[i+1,j].x,1)), color='white', fontsize=8, bbox={'facecolor': 'red', 'alpha': 0.6, 'pad': 2})


    # -----plotting the info about the clients------
    ax.plot((coord_clients[:,1]), (coord_clients[:,0]), 'wo') #clients as white dots

    for i in range(len(coord_clients)):
        ax.text((coord_clients[i,1]), (coord_clients[i,0]) - 0.001, str(i+1+len(P)), color='white', fontsize=10,
                bbox={'facecolor': 'red', 'alpha': 0.6, 'pad': 2})
        if (print_tau == True):
            for j in K:
                if tau[i+1+len(P),j].x:
                    ax.text((coord_clients[i,1]), (coord_clients[i,0])-0.002, r'$\tau$' + "=" + str(round(tau[i+1+len(P),j].x,1)), color='white', fontsize=8, bbox={'facecolor': 'red', 'alpha': 0.6, 'pad': 2})

    plt.xlabel("Longitutde (" + u"\N{DEGREE SIGN}" + "E)")
    plt.ylabel("Latitude (" + u"\N{DEGREE SIGN}" + "N)")
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