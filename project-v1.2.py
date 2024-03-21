
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

### PRE-USER INPUT
## Saturated Vapour Pressure function
def pvs(Temp): 
    '''Saturated Vapour Pressure [Pa]'''
    pvs = np.power(10, (2.7877+7.625*(Temp/(241.6+Temp)))); return pvs

## Extracting Layer Data
LayerData = pd.read_csv('projectData/LayerData.csv', index_col=0)

### USER INPUT:

## Number of layers:
n = int(input("Please enter the number of layers: ")) 
print()

## Initialising layer property arrays:
layer_data = []
thck = [0]*n                    # Thickness [m]
ID = [0]*n                      # Layer ID

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
    # following based on internal requirements

''' The following variables are not directly assigned in the df
    as they will depend on user input and meteostat data in a future version. '''

hc_i = 1/0.14               # convection coefficient [W/m²K]
hm_i = 1.11E-3              # mass transfer coefficient []
internal_Temperature = 20   #!#
hum_i = 0.5                 #!#
pvs_i = pvs(internal_Temperature)
pv_i =  pvs_i*hum_i

## External conditions: 
# '#!#'; to depend on méteostat
hc_e = 1/0.04               #!#
hm_e = 1.06E-2              #!#
external_Temperature = 0    #!#
hum_e = 0.9                 #!#
pvs_e = pvs(external_Temperature)
pv_e =  pvs_e*hum_e

## Constructing DataFrame with Internal/External Conditions
internal_conds = [hc_i, hm_i, internal_Temperature, hum_i, pvs_i, pv_i]
external_conds = [hc_e, hm_e, external_Temperature, hum_e, pvs_e, pv_e]
Conditions_row_names = ['Internal', 'External']
Conditions_column_names = ['Convection', 'MassTransfer', 'Temperature', 'RelativeHumidity', 'SatVapPressure', 'VapPressure']

Conditions = pd.DataFrame([internal_conds, external_conds], index=Conditions_row_names, columns=Conditions_column_names)

## Assigning layer information to a dataframe:

# Appending all layer data to a list:
for layer in range(0,n):
    layer_data.append([
        LayerData.index[ID[layer]],                     # Layer Name
        thck[layer],                                    # Thickness [m]
        LayerData["k"].iloc[ID[layer]],                 # Linear heat transfer coefficient [W/m/K]
        LayerData["pi"].iloc[ID[layer]],                # Permeability coefficient  [kg/m/s/Pa]
        thck[layer]/LayerData["k"].iloc[ID[layer]],     # Thermal Resistance [m²*K/W]
        thck[layer]/LayerData["pi"].iloc[ID[layer]]     # (Vapour) Resistance [m²*K/W]
        ])

# Converting list to dataframe:
Layers_row_names = ['LAYER {}'.format(i+1) for i in range(0,n)]
Layers_column_names = ['LayerName', 'Thickness', 'k', 'Permeability', 'Rth', 'Rv']

Layers = pd.DataFrame(layer_data, columns=Layers_column_names)
Layers.columns = Layers_column_names
Layers.index = Layers_row_names

ghd = sum(Layers['Rth'])

## Heat and Moisture Transfer
Rth_total = (1/Conditions.Convection['Internal']) + sum(Layers['Rth']) + (1/Conditions.Convection['External'])
thermal_flow = (1/Rth_total) * (Conditions.Temperature['Internal']-Conditions.Temperature['External']) 
Rv_total = (1/Conditions.MassTransfer['Internal']) + sum(Layers['Rv']) + (1/Conditions.MassTransfer['External'])

### TEMPERATURE EVOLUTION
Mag = np.floor(np.log10(thck_tot))
ticks = (10**Mag)/100; # assures the graph will always have the correct amount of ticks
# ticks = 0.001;
iterations = int(-((-thck_tot/ticks)//1)); #rounds number and converts to integer

x_real = [sum(thck[:i+1]) for i in range(n)] # array indicating each layer boundary
x = [(sum(thck[:i+1])/ticks) for i in range(n)] # layer boundaries in terms of number of iterations

Temperature = []; Temperature.append(Conditions.Temperature['Internal']-(thermal_flow/Conditions.Convection['Internal'])) #initial value

for i in range(1, iterations):
    if (n>0) and (i <= x[0]): Temperature.append(Temperature[i-1]-(thermal_flow*(ticks/Layers.k['LAYER 1'])))
    elif (n>1) and (x[0] < i) and (i <= x[1]): Temperature.append(Temperature[i-1]-(thermal_flow*(ticks/Layers.k['LAYER 2'])))
    elif (n>2) and (x[1] < i) and (i <= x[2]): Temperature.append(Temperature[i-1]-(thermal_flow*(ticks/Layers.k['LAYER 3'])))

### PRESSURE EVOLUTIONS


## Initialising pressure arrays
pvs_vals = []; pvs_vals.append(Conditions.SatVapPressure['Internal'])
pv_vals = []; pv_vals.append(Conditions.VapPressure['Internal'])

## Assigning external pressures:
# pvs_e = pvs(external_Temperature)
# pv_e = hum_e*pvs_e

## Calculating vapour flow through the wall:
vapour_flow=(1/Rv_total)*(pv_vals[0]-Conditions.SatVapPressure['External'])

for i in range(1, iterations):
    pvs_vals.append(pvs(Temperature[i]))

    if (n>0) and (i <= x[0]): pv_vals.append(pv_vals[i-1]-(vapour_flow*(ticks/Layers.Permeability['LAYER 1'])))
    elif (n>1) and (x[0] < i) and (i <= x[1]): pv_vals.append(pv_vals[i-1]-(vapour_flow*(ticks/Layers.Permeability['LAYER 2'])))
    elif (n>2) and (x[1] < i) and (i <= x[2]): pv_vals.append(pv_vals[i-1]-(vapour_flow*(ticks/Layers.Permeability['LAYER 3'])))
    

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
    # ax1.xaxis.set_label_coords(np.mean([x[i-1], x[i]]), 1)
    # ax1.set_xlabel(Layer_Material[i-1])

ax1.legend(loc='upper center', fontsize='x-small')
ax2.legend(loc='upper right', fontsize='x-small')
plt.show()

# for i in range(1,n):
#     x_label_position = np.mean([x_real[i - 1], x_real[i]])
#     y_label_position = max(Temperature) + 0.1
#     plt.text(x_label_position, y_label_position, Layers.at[i-1, 'Layer Name'], ha='center')

# plt.tight_layout()

### VAPOUR BARRIER SUGGESTION
# VapourBarrierData = pd.read_csv('projectData/VapourBarrierData', index_col=0)

# VB = []
