# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 14:03:51 2020

@author: Flori
"""

import numpy as np
import xlrd
from geopy.distance import distance
import matplotlib.pyplot as plt
import csv

#defining the location of the airbase, pizzerias, clients
loc_airbase=np.array((52.020949, 4.393304))#latitude longitude
loc_pizzerias=np.array([(52.012892, 4.373290),(52.013430, 4.363525),(52.014091, 4.359047),
                        (52.011641, 4.358145),(52.001018, 4.354525),(51.999181, 4.355258),
                        (52.004122, 4.35135), (51.996050, 4.375586), (52.00577, 4.3784140), (52.0008320, 4.376571)])

pizzeria_expected_arrival=np.array([600, 750, 800, 760, 950, 820, 780, 680, 1000, 820])

loc_clients=np.array([(51.987226, 4.349195),(51.988648, 4.348359),(51.990306, 4.353927),
                      (51.989985, 4.376466),(52.010301, 4.348025), (52.021676, 4.356424),
                      (52.012930, 4.361411), (52.014263, 4.359737), (52.02149112433499, 4.349385881647885),
                      (52.02092781375249, 4.36390154038701), (52.01947926825116, 4.378417199126135), (52.01505286635093, 4.359324530874673),
                      (52.00821120194048, 4.359847617676083), (52.014006561907664, 4.361547649780666), (52.010787010146814, 4.3891404785550385),
                      (52.005071735222664, 4.359455302575026), (52.00531324048147, 4.372140157509216), (52.00402519737562, 4.390971282359972),
                      (51.997262362906916, 4.358932215773615), (51.999114192971916, 4.370309353704281), (51.991464834414685, 4.353047489257754),
                      (51.991786939025324, 4.355401379864099), (52.022295841432594, 4.357886042170796), (52.002231917610345, 4.372035277875838),
                      (51.99735984356062, 4.364269669691543), (52.021941621223, 4.375752949688681), (52.01482576386368, 4.350198149585744),
                      (52.005376069601844, 4.378765112438456), (52.00256469949602, 4.380125444002871), (52.01416793838498, 4.3614694682623245)])

customer_arrival_time= np.array([(700, 1300), (700, 1400), (700,1500),
                                 (700, 1600), (700, 1700), (750, 1400),
                                 (750, 1300), (750, 1500), (750, 1400),
                                 (800, 1250), (800, 1600), (800, 1590),
                                 (800, 1700), (800, 1750), (850, 1400),
                                 (850, 1550), (850, 1600), (850, 1800),
                                 (900, 1850), (900, 1670), (900, 1700),
                                 (900, 1900), (950, 1500), (950, 1800),
                                 (950, 1780), (950, 1860), (1000, 1450),
                                 (1000,1900), (1000, 1700), (1000, 1890)])

#++++++++++++++++++ Generating the "client_1_to_client_2_distances" file ++++++++++++++++++++++++

myData_client_1_client_2_distances = [['latitude_client_1','longitude_client_1','latitude_client_2','longitude_client_2','distance']]
for client_1 in loc_clients:
    for client_2 in loc_clients:
        if client_1[0]!=client_2[0] and client_1[1]!=client_2[1]:
            myData_client_1_client_2_distances+=[[round(client_1[0],5),round(client_1[1],5),round(client_2[0],5),round(client_2[1],5),round(distance(client_1,client_2).m,4)]]

myFile_client_1_client_2_distances = open('client_1_client_2_distances_complex.csv', 'w', newline="")
with myFile_client_1_client_2_distances:
   writer = csv.writer(myFile_client_1_client_2_distances)
   writer.writerows(myData_client_1_client_2_distances)


#++++++++++++++++++ Generating the "client_airbase_distances" file ++++++++++++++++++++++++
myData_client_airbase_distances=[['latitude_client','longitude_client','latitude_airbase','longitude_airbase','distance']]
for client in loc_clients:
    myData_client_airbase_distances+=[[round(client[0],5), round(client[1],5),round(loc_airbase[0],5),round(loc_airbase[1],5),
                                       round(distance(client,loc_airbase).m,5)]]
    
myFile_client_airbase_distances = open('client_airbase_distances_complex.csv', 'w', newline="")
with myFile_client_airbase_distances:
   writer = csv.writer(myFile_client_airbase_distances)
   writer.writerows(myData_client_airbase_distances)


#++++++++++++++++++ Generating the "pizzeria_expected_arrival_time" file ++++++++++++++++++++++++
myData_pizzeria_expected_arrival_time=[['latitude_pizzeria','longitude_pizzeria','expected_arrival_time(in sec)']]
i=0
for pizzeria in loc_pizzerias:
    myData_pizzeria_expected_arrival_time+=[[round(pizzeria[0],5),round(pizzeria[1],5),round(pizzeria_expected_arrival[i],0)]]
    i+=1
    
myFile_pizzeria_expected_arrival_time = open('pizzeria_expected_arrival_time_complex.csv', 'w', newline="")
with myFile_pizzeria_expected_arrival_time:
   writer = csv.writer(myFile_pizzeria_expected_arrival_time)
   writer.writerows(myData_pizzeria_expected_arrival_time)

#++++++++++++++++++ Generating the "airbase_pizzerias_distances" file ++++++++++++++++++++++++
myData_airbase_pizzerias = [['latitude_airbase','longitude_airbase','latitude_pizzeria','longitude_pizzeria','distance']]

for pizzeria in loc_pizzerias:
    myData_airbase_pizzerias += [[round(loc_airbase[0], 5), round(loc_airbase[1], 5),
                                  round(pizzeria[0], 5), round(pizzeria[1], 5), round(distance(loc_airbase, pizzeria).m, 5)]]

myFile_airbase_pizzerias = open('airbase_pizzerias_distances_complex.csv', 'w', newline="")
with myFile_airbase_pizzerias:
    writer = csv.writer(myFile_airbase_pizzerias)
    writer.writerows(myData_airbase_pizzerias)

#++++++++++++++++++ Generating the "pizzerias_clients_distances" file ++++++++++++++++++++++++
myData_pizzerias_clients = [['latitude_pizzeria','longitude_pizzeria','latitude_client','longitude_client','distance']]

for pizzeria in loc_pizzerias:
    for client in loc_clients:
        myData_pizzerias_clients += [[round(pizzeria[0], 5), round(pizzeria[1], 5),
                                  round(client[0], 5), round(client[1], 5), round(distance(pizzeria, client).m, 5)]]

myFile_pizzerias_clients = open('pizzerias_clients_complex.csv', 'w', newline="")
with myFile_pizzerias_clients:
    writer = csv.writer(myFile_pizzerias_clients)
    writer.writerows(myData_pizzerias_clients)

#++++++++++++++++++ Generating the "customer_arrival_time" file ++++++++++++++++++++++++
myData_customer_arrival_time = [['lower_bound_time_interval','upper_bound_time_interval']]

for customer_time_bounds in customer_arrival_time:
    myData_customer_arrival_time += [[customer_time_bounds[0],customer_time_bounds[1]]]

myFile_customer_arrival_time = open('customer_arrival_time_complex.csv', 'w', newline="")
with myFile_customer_arrival_time:
    writer = csv.writer(myFile_customer_arrival_time)
    writer.writerows(myData_customer_arrival_time)