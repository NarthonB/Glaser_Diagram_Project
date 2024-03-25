
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

### PRE-USER INPUT

## All equations used in the code: (Descriptions written in triple quotes)
def pvs(Temp): 
    '''Saturated Vapour Pressure [Pa]'''
    return np.power(10, (2.7877 + 7.625 * (Temp/(241.6+Temp))))
    
def convection_coeff(v):
    '''Convection Coefficient 
    (https://www.engineeringtoolbox.com/convective-heat-transfer-d_430.html)'''
    hc = 10.45 - v + 10*np.power(v,0.5); return hc

def variable_evolution(var_prev, flow, conductivity, ticks):
    '''Calculates a variable's (temperature or vapour pressure) evolution through the wall'''
    return var_prev - flow*(ticks/conductivity)

def determine_layer_num(layer_bounds):
    '''Determines which layer a variable is in'''
    n = len(layer_bounds)
    if ((n>0) and (i <= layer_bounds[0])): return 1
    elif ((n>1) and (layer_bounds[0] < i)) and (i <= layer_bounds[1]): return 2
    elif ((n>2) and (layer_bounds[1] < i) and (i <= layer_bounds[2])): return 3
    
def Total_resistance(int_transfer, wall_resistance, ext_transfer):
    '''Calculates the total thermal/vapour resistance of the wall including internal and external resistances'''
    return 1/int_transfer + sum(wall_resistance) + 1/ext_transfer

def Total_flow(R_tot, int_cond, ext_cond):
    '''Calculates the total thermal or vapour flow through the wall'''
    return (1/R_tot)*(int_cond - ext_cond)

def avg(a,b):
    '''Simple averaging function'''
    return (a+b)/2

## Extracting Layer Data
MaterialData = pd.read_csv('projectData/LayerData.csv', index_col=0)

### USER INPUT:

## Number of layers:
n = int(input("Please enter the number of layers: ")) 
print()

## Initialising layer property arrays:
layer_data = []
thck = [0]*n        # Thickness [m]
ID = [0]*n          # Layer ID

##  Collecting individual layer data from user:
for i in range(n):
    ("-" * 50)
    print(f"Let's get data for layer {i+1}:")
    print("Here is the choice of materials: \n")
    print(MaterialData); print()
    ID[i] = int(input(f"Insert the ID number of your material for layer {i + 1}: "))
    thck[i] = float(input(f"Layer thickness [m] for layer {i + 1}: "))
        
    print("-" * 50, end="\n")


### CALCULATION OF CONSTANT VALUES

# Total thickness
thck_tot = sum(thck)

## Assigning Properties for each layer:

''' The following variables are not assigned directly into the df
    as they will depend on user input and Meteostat data in a future version. '''

# Internal conditions: (will depend on user's internal requirements)
air_speed_i = 0.2
hc_i = convection_coeff(air_speed_i)      # convection coefficient [W/m²K]
hm_i = 1.11E-3                           # mass transfer coefficient []
internal_Temperature = 20   #!#
hum_i = 0.5                 #!#
pvs_i = pvs(internal_Temperature)
pv_i =  pvs_i*hum_i

# External conditions: 
# '#!#'; to depend on méteostat
air_speed_e = 4
hc_e = convection_coeff(air_speed_e)               #!#
hm_e = 1.06E-2              #!#
external_Temperature = 0    #!#
hum_e = 0.9                 #!#
pvs_e = pvs(external_Temperature)
pv_e =  pvs_e*hum_e

# Constructing DataFrame with Internal/External Conditions
internal_conds = [hc_i, hm_i, internal_Temperature, hum_i, pvs_i, pv_i]
external_conds = [hc_e, hm_e, external_Temperature, hum_e, pvs_e, pv_e]
Conditions_row_names = ['Internal', 'External']
Conditions_column_names = ['Convection', 'MassTransfer', 'Temperature', 'RelativeHumidity', 'SatVapPressure', 'VapPressure']

Conditions = pd.DataFrame([internal_conds, external_conds], index=Conditions_row_names, columns=Conditions_column_names)

## Assigning layer information to a dataframe:

# Appending all layer data to a list:
for layer in range(0,n):
    layer_data.append([
        MaterialData.index[ID[layer]],                     # Layer Name
        thck[layer],                                    # Thickness [m]
        MaterialData["k"].iloc[ID[layer]],                 # Linear heat transfer coefficient [W/m/K]
        MaterialData["pi"].iloc[ID[layer]],                # Permeability coefficient  [kg/m/s/Pa]
        thck[layer]/MaterialData["k"].iloc[ID[layer]],     # Thermal Resistance [m²*K/W]
        thck[layer]/MaterialData["pi"].iloc[ID[layer]]     # (Vapour) Resistance [m²*K/W]
        ])

# Converting list to dataframe:

Layer_Props = pd.DataFrame(layer_data)
Layer_Props.columns = ['LayerName', 'Thickness', 'k', 'Permeability', 'Rth', 'Rv']
Layer_Props.index = ['LAYER {}'.format(i+1) for i in range(0,n)]

## Total resistances and thermal flow

# Thermal and Vapour Resistances  
Rth_total = Total_resistance(Conditions.Convection['Internal'], Layer_Props['Rth'], Conditions.Convection['External'])
Rv_total = Total_resistance(Conditions.MassTransfer['Internal'], Layer_Props['Rv'], Conditions.MassTransfer['External'])
# Total thermal flow
thermal_flow = Total_flow(Rth_total, Conditions.Temperature['Internal'], Conditions.Temperature['External']) 

### TEMPERATURE EVOLUTION

# Setting number of ticks and iterations
Magnitude = np.floor(np.log10(thck_tot))    # determine the order of magnitude
ticks = (10**Magnitude)/100;                # assures the graph will always have the correct amount of ticks
# ticks = 0.001;
iterations = int(-((-thck_tot/ticks)//1));  #rounds number and converts to integer

# Arrays containing layer boundaries
x_real = [sum(thck[:i+1]) for i in range(n)]    # array indicating each layer boundary
x = [(sum(thck[:i+1])/ticks) for i in range(n)] # layer boundaries in terms of number of iterations

## Calculating and storing Temperature Evolution in an array  

# Initial Temperature Value
Temperature = [Conditions.Temperature['Internal']-(thermal_flow/Conditions.Convection['Internal'])]
# Filling array with a loop
for i in range(1, iterations):
    Temperature.append(variable_evolution(Temperature[i-1], thermal_flow, Layer_Props.k[f"LAYER {determine_layer_num(x)}"], ticks))


### PRESSURE EVOLUTIONS

# Initialising pressure arrays
pvs_vals = [Conditions.SatVapPressure['Internal']]
pv_vals = [Conditions.VapPressure['Internal']]

## Calculating vapour flow through the wall:
vapour_flow = Total_flow(Rv_total, pv_vals[0], Conditions.SatVapPressure['External'])

## Calculating and storing Pressure evolution in arrays
# Filling arrays with a loop
for i in range(1, iterations):
    pvs_vals.append(pvs(Temperature[i]))
    pv_vals.append(variable_evolution(pv_vals[i-1], vapour_flow, Layer_Props.Permeability[f"LAYER {determine_layer_num(x)}"], ticks))


### LIST CONVERSION

Temperature = np.array(Temperature)
pv_vals = np.array(pv_vals)
pvs_vals = np.array(pvs_vals)
x_vals = np.arange(0, thck_tot, ticks); x_vals = np.array(x_vals)
condensation_point = [];

### FINAL CALCULATIONS

## Finding condensation point:
check = True;
for i in range(1, iterations):
    if (np.mean(pv_vals[i-1:i]) >= np.mean(pvs_vals[i-1:i])) and (check==True): # checks if pv>pvs in each iteration
        condensation_point.append(avg(i-1, i)*ticks)        # stores condensation point in the array
        check = False                                       # stops the loop checking once point is located
    if (np.mean(pv_vals[i-1:i]) <= np.mean(pvs_vals[i-1:i])): check = True  # allows loop to start checking after first area of condensation

print ("condenstation point = ", condensation_point)


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

