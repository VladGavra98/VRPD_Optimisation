# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 17:47:35 2020


The purpose is (well) to do the sensitivity analysis.


@author: vladg,danny,flori
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as mpath


star = mpath.Path.unit_regular_star(6)
circle = mpath.Path.unit_circle()

# Plotting:
texpsize = [15, 18, 22]

plt.rc('font', size=texpsize[1], family='serif')  # controls default text sizes
plt.rc('axes', titlesize=texpsize[1])  # fontsize of the axes title
plt.rc('axes', labelsize=texpsize[1])  # fontsize of the x and y labels
plt.rc('xtick', labelsize=texpsize[0])  # fontsize of the tick labels
plt.rc('ytick', labelsize=texpsize[0])  # fontsize of the tick labels
plt.rc('legend', fontsize=texpsize[0])  # legend fontsize
plt.rc('figure', titlesize=texpsize[2])  # fontsize of the figure title
matplotlib.rcParams['lines.linewidth'] = 1.5
matplotlib.rcParams['figure.facecolor'] = 'white'
matplotlib.rcParams['axes.facecolor'] = 'white'
matplotlib.rcParams["legend.fancybox"] = False





    # CHANGE HERE WITH YOUR OWN DATA!
q_lst = [8,10,12,14,16]
Z1_lst = [31384,31384,31384,31384,31384]
Z2_lst = [3138.4,3138.4,3138.4,3138.4,3138.4]
Z3_lst = [17443.5,17443.5,17443.5,17443.5,17443.5]

#Z1
plt.figure("Z1_q")

x  = np.array(q_lst)
y  = np.array(Z1_lst)/1000      #to get km
ax1 = plt.subplot(111)


plt.plot(x,y,"#2A9D8F",marker='s', label=str(r"$Z_3$"))
plt.xlabel(r'$q$ [pizzas]')
plt.ylabel(r'$Z_1$ [km]')
plt.title(r'Minimised Total Distance')

ax1.set_xlim(min(x), max(x))
ax1.set_ylim(0.9*min(y), 1.1*max(y))
plt.axvline(0,color=(0,0,0),linewidth=0.8) #comment out for no axes line
plt.axhline(0,color=(0,0,0),linewidth=0.8) #comment out for no axes line
plt.minorticks_on() # set minor ticks
plt.grid(which='major', linestyle='-', linewidth='0.3', color='black') # customise major grid
plt.grid(which='minor', linestyle=':', linewidth='0.3', color='grey') # customise minor grid
plt.legend()


plt.tight_layout()
plt.show()


# Z2
plt.figure("Z2_q")
y  = np.array(Z2_lst)/60     #to get km
ax2 = plt.subplot(111)



plt.plot(x,y,"#227C9D",marker='s', label=str(r"$Z_3$"))
plt.xlabel(r'$q$ [pizzas]')
plt.ylabel(r'$Z_2$ [min]')
plt.title(r'Minimised Total Airtime')

ax2.set_xlim(min(x), max(x))
ax2.set_ylim(0.9*min(y), 1.1*max(y))

plt.axvline(0,color=(0,0,0),linewidth=0.8) #comment out for no axes line
plt.axhline(0,color=(0,0,0),linewidth=0.8) #comment out for no axes line
plt.minorticks_on() # set minor ticks
plt.grid(which='major', linestyle='-', linewidth='0.3', color='black') # customise major grid
plt.grid(which='minor', linestyle=':', linewidth='0.3', color='grey') # customise minor grid
plt.legend()

plt.tight_layout()
plt.show()



#Z3
plt.figure("Z3_q")
y  = np.array(Z3_lst)/60     #to get km
ax3 = plt.subplot(111)

plt.plot(x,y,"#E76F51",marker='s', label=str(r"$Z_3$"))
plt.xlabel(r'$q$ [pizzas]')
plt.ylabel(r'$Z_3$ [min]')
plt.title(r'Minimised Total Lateness')

ax3.set_xlim(min(x), max(x))
ax3.set_ylim(0.9*min(y), 1.1*max(y))

plt.axvline(0,color=(0,0,0),linewidth=0.8) #comment out for no axes line
plt.axhline(0,color=(0,0,0),linewidth=0.8) #comment out for no axes line
plt.minorticks_on() # set minor ticks
plt.grid(which='major', linestyle='-', linewidth='0.3', color='black') # customise major grid
plt.grid(which='minor', linestyle=':', linewidth='0.3', color='grey') # customise minor grid
plt.legend()



plt.tight_layout()
plt.show()

