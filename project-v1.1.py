import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

### PRE-USER INPUT
LayerData = pd.read_csv('projectData/LayerData.csv', index_col=0)

### USER INPUT:

## Number of layers:
n = int(input("Please enter the number of layers: ")) 
print()

## Initialising layer properties:
layer_data = []
thck = [0]*n; ID = [0]*n    # Thickness [m]
k = [0]*n                   # Linear heat transfer coefficient [W/m/K]
permeability = [0]*n        # Permeability coefficient [kg/m/s/Pa]
Rth = [0]*n                 # Thermal Resistance [m²*K/W]
Rv = [0]*n                  # (Vapour) Resistance [Pa*s/kg]

##  Collecting individual layer data from user:
for i in range(n):
    ("-" * 50)
    print("Let's get data for layer", i+1,":")
    print("Here is the choice of materials: \n")
    print(LayerData); print()
    ID[i] = int(input(f"Insert the ID number of your material for layer {i + 1}: "))
    thck[i] = float(input(f"Layer thickness [m] for layer {i + 1}: "))
        
    print("-" * 50, end="\n")


thck_tot = sum(thck)

# Assigning Properties for each layer:

## Internal conditions:
hc_i = 1/0.14      # convection coefficient [W/m²K]
hm_i = 1.11E-3     #  mass transfer coefficient []

    # following based on internal requirements
internal_Temperature = 20
hum_i = 0.5

## Assigning layer properties:
for layer in range(0,n):
    k[layer] = LayerData["k"].iloc[ID[layer]]
    permeability[layer] = LayerData["pi"].iloc[ID[layer]]
    Rth[layer] = thck[layer]/k[layer]
    Rv[layer] = thck[layer]/permeability[layer]

## External conditions: 
hc_e = 1/0.04
hm_e = 1.06E-2

#!# following to depend on méteostat:
T_e = 0
hum_e = 0.9


## Heat and Moisture Transfer
Rth_tot = (1/hc_i) + sum(Rth) + (1/hc_e)
thermal_flow = (1/Rth_tot) * (internal_Temperature-T_e) 
Rv_total = (1/hm_i)+ sum(Rv) +(1/hm_e)

### TEMPERATURE EVOLUTION
Mag = np.floor(np.log10(thck_tot))
ticks = (10**Mag)/100; # assures the graph will always have the correct amount of ticks
# ticks = 0.001;
iterations = int(-((-thck_tot/ticks)//1)); #rounds number and converts to integer

x_real = [sum(thck[:i+1]) for i in range(n)] # array indicating each layer boundary
x = [(sum(thck[:i+1])/ticks) for i in range(n)] # layer boundaries in terms of number of iterations

Temperature = []; Temperature.append(internal_Temperature-(thermal_flow/hc_i)) #initial value

for i in range(1, iterations):
    if (n>0) and (i <= x[0]): Temperature.append(Temperature[i-1]-(thermal_flow*(ticks/k[0])))
    elif (n>1) and (x[0] < i) and (i <= x[1]): Temperature.append(Temperature[i-1]-(thermal_flow*(ticks/k[1])))
    elif (n>2) and (x[1] < i) and (i <= x[2]): Temperature.append(Temperature[i-1]-(thermal_flow*(ticks/k[2])))

### PRESSURE EVOLUTIONS
def pvs(Temp): 
    '''Saturated Vapour Pressure [Pa]'''
    pvs = np.power(10, (2.7877+7.625*(Temp/(241.6+Temp)))); return pvs

## Initialising pressure arrays
pvs_vals = []; pvs_vals.append(pvs(Temperature[0]))
pv_vals = []; pv_vals.append(hum_i*pvs_vals[0])

## Assigning external pressures:
pvs_e = pvs(T_e)
pv_e = hum_e*pvs_e

## Calculating vapour flow through the wall:
vapour_flow=(1/Rv_total)*(pv_vals[0]-pv_e)

for i in range(1, iterations):
    pvs_vals.append(pvs(Temperature[i]))

    if (n>0) and (i <= x[0]): pv_vals.append(pv_vals[i-1]-(vapour_flow*(ticks/permeability[0]))) #!#
    elif (n>1) and (x[0] < i) and (i <= x[1]): pv_vals.append(pv_vals[i-1]-(vapour_flow*(ticks/permeability[1]))) #!#
    elif (n>2) and (x[1] < i) and (i <= x[2]): pv_vals.append(pv_vals[i-1]-(vapour_flow*(ticks/permeability[2]))) #!#
    

### LIST CONVERSION
Temperature = np.array(Temperature)
pv_vals = np.array(pv_vals)
pvs_vals = np.array(pvs_vals)
x_vals = np.arange(0, thck_tot, ticks); x_vals = np.array(x_vals)

### PLOTTING
fig, ax1 = plt.subplots()

## Temperature evolution:
ax1.plot(x_vals, Temperature, '#F62817', label='Temperature [°C]', linewidth=0.85)
ax1.set_xlabel('Position [m]')
ax1.set_ylabel('Temperature [°C]', color='k')
ax1.tick_params('y', colors='k')

## Displaying layer boundaries 
if n>1: ax1.axvline(x=x[0]*ticks, color='k', linestyle='-', linewidth=0.95) #!#
if n>2: ax1.axvline(x=x[1]*ticks, color='k', linestyle='-', linewidth=0.95) #!#

## Saturated Pressure Evolution (includes adding an extra y-axis):
ax2 = ax1.twinx()
ax2.plot(x_vals, pvs_vals, 'b', linestyle='--', label='pvs', linewidth=0.95)
ax2.set_ylim(0, (max(pv_vals)+400))
ax2.set_ylabel('pvs', color='k')
ax2.tick_params('y', colors='k')

## Pressure Evolution:
ax2.plot(x_vals, pv_vals, 'cyan', linestyle='--', label='pv', linewidth=0.95)

## Highlighting condensation area
ax2.fill_between(x_vals, pv_vals, pvs_vals, where=(pv_vals >= pvs_vals), color='blue', alpha=0.2)

## Finalizing and displaying of plot
ax1.legend(loc='upper center')
ax2.legend(loc='upper right')
plt.show()