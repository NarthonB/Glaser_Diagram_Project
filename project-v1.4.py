
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

###! PRE-USER INPUT

## All equations used in the code: (Descriptions written in triple quotes)

def pvs(Temperature): 
    '''Saturated Vapour Pressure [Pa]'''
    return np.power(10, (2.7877 + 7.625 * (Temperature/(241.6+Temperature))))
    
def convection_coeff(v):
    '''Convection Coefficient 
    (https://www.engineeringtoolbox.com/convective-heat-transfer-d_430.html)'''
    hc = 10.45 - v + 10*np.power(v,0.5); return hc

def variable_evolution(var_prev, flow, conductivity, ticks):
    '''Calculates a variable's (temperature or vapour pressure) evolution through the wall'''
    return var_prev - flow*(ticks/conductivity)

def determine_layer_num(x, layer_bounds):
    '''Determines which layer a variable is in'''
    n = len(layer_bounds) - 1
    if ((n>0) and (x <= layer_bounds[1])): return int(1)
    elif ((n>1) and (layer_bounds[1] < x)) and (x <= layer_bounds[2]): return int(2)
    elif ((n>2) and (layer_bounds[2] < x) and (x <= layer_bounds[3])): return int(3)
    
def Total_resistance(int_transfer, wall_resistance, ext_transfer, VB_permeance):
    '''Calculates the total thermal/vapour resistance of the wall including internal and external resistances'''
    if (VB_permeance>0): return 1/int_transfer + sum(wall_resistance) + 1/ext_transfer + 1/VB_permeance
    else: return 1/int_transfer + sum(wall_resistance) + 1/ext_transfer

def Total_flow(R_tot, int_cond, ext_cond):
    '''Calculates the total thermal or vapour flow through the wall'''
    return (1/R_tot)*(int_cond - ext_cond)

# def on_plot_hover(event):
#     # Iterating over each data member plotted
#     for curve in plot.get_lines():
#         # Searching which data member corresponds to current mouse position
#         if curve.contains(event)[0]:
#             print("over %s" % curve.get_gid())

def avg(a,b):
    '''Simple averaging function'''
    return (a+b)/2


def colourpicker(Material_Name):
    '''Assigns colour values to Materials'''
    if Material_Name == 'Glass Wool': return '#b79d44'
    elif Material_Name == 'Concrete': return '#848484'
    elif Material_Name == 'Brick': return '#a04440'
    elif Material_Name == 'Expanded Polystyrene': return '#ffffff'
    elif Material_Name == 'Wood': return '#cc884c'
    elif Material_Name == 'Plaster': return '#90908c'
    else: return 'white'

def rounduptoint(number):
    '''Rounds up float values to nearest integer'''
    return int(-((-number)//1))

def dfplot(df, layer_boundaries):
    '''Takes in a dataframe with the following columns:
    {x values (indexed), Temperature, SatVapPressure, VapPressure, Material Names}
    as well as a list of layer separations
    and plots a Glaser Diagram to show the condensation
    '''
    # Converting df columns to arrays 
    x_vals = np.array(df.index)
    Temperature = np.array(df.iloc[:,0])
    pvs_vals = np.array(df.iloc[:,1])
    pv_vals = np.array(df.iloc[:,2])
    
    ticks = np.diff(x_vals)[0]  # calculates ticks
    # layerbounds = [(sum(thck[:i+1])/ticks) for i in range(len(layer_boundaries))]
    
    fig, ax1 = plt.subplots()

    ## Temperature evolution:
    ax1.plot(x_vals, Temperature, '#F62817', label='Temperature [°C]', linewidth=0.85)
    ax1.set_xlabel('Position [m]')
    ax1.set_ylabel('Temperature [°C]', color='k')
    ax1.tick_params('y', colors='k')

    ## Displaying layer boundaries 
    for i in range(0, len(layer_boundaries)):
        if (i==0) or (i==len(layer_boundaries)):
            ax1.axvline(x=layer_boundaries[i], color='k', linestyle='-', linewidth=1.3)
        else: ax1.axvline(x=layer_boundaries[i], color='k', linestyle='-', linewidth=1)
        if i>0:ax1.axvspan(
            layer_boundaries[i-1], layer_boundaries[i],
            color=colourpicker(df.loc[layer_boundaries[i],'Material']))
            
    ## Saturated Pressure Evolution (includes adding an extra y-axis):
    ax2 = ax1.twinx()
    ax2.plot(x_vals, pvs_vals, 'b', linestyle='--', label='pvs', linewidth=0.95)
    ax2.set_ylim(0, (max(pv_vals)+400))
    ax2.set_ylabel('Pressure [Pa]', color='k')
    ax2.tick_params('y', colors='k')

    ## Pressure Evolution:
    ax2.plot(x_vals, pv_vals, 'cyan', linestyle='--', label='pv', linewidth=0.95)

    ## Highlighting condensation area
    ax2.fill_between(x_vals, pv_vals, pvs_vals, where=(pv_vals >= pvs_vals), color='cyan', alpha=0.2)

    ## Finalizing and displaying of plot
        # ax1.xaxis.set_label_coords(np.mean([x[i-1], x[i]]), 1)
        # ax1.set_xlabel(Layer_Material[i-1])

    ax1.legend(loc='upper center', fontsize='x-small')
    ax2.legend(loc='upper right', fontsize='x-small')



## Extracting Layer Data
MaterialData = pd.read_csv('projectData/LayerData.csv', index_col=0)
# MaterialData['pi'] = MaterialData['pi'].apply(lambda x: format(x, '0.3g'))
# MaterialData.style
# MaterialData.style.format({'pi': formatdfcols})

###! USER INPUT:

## Number of layers:
n = int(input("Please enter the number of layers: ")) 
print()

## Initialising layer property lists:
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


###! CALCULATION OF CONSTANT VALUES

# Total thickness
thck_tot = sum(thck)

## Assigning Properties for each layer:

''' The following variables are not assigned directly into the df
    as they will depend on user input and Meteostat data in a future version. '''

# Internal conditions: (#*# will depend on user's internal requirements)
air_speed_i = 0.2                       #*#
hc_i = convection_coeff(air_speed_i)        # convection coefficient [W/m²K]
# hm_i = 1.11e-3                            # mass transfer coefficient []
hm_i = hc_i*6.9e-9
internal_Temperature = 20               #*#
hum_i = 0.5                             #*#
pvs_i = pvs(internal_Temperature)
pv_i =  pvs_i*hum_i

# External conditions: (#*# to depend on méteostat)
air_speed_e = 4                         #*#
hc_e = convection_coeff(air_speed_e)    #*#
# hm_e = 1.06E-2
hm_e = hc_e*6.9e-9
external_Temperature = 0                #*#
hum_e = 0.9                             #*#
pvs_e = pvs(external_Temperature)
pv_e =  pvs_e*hum_e

# Constructing DataFrame with Internal/External Conditions
Conditions = pd.DataFrame(
    [[hc_i, hm_i, internal_Temperature, hum_i, pvs_i, pv_i],
     [hc_e, hm_e, external_Temperature, hum_e, pvs_e, pv_e]],
    columns=['Convection', 'MassTransfer', 'Temperature', 'RelativeHumidity', 'SatVapPressure', 'VapPressure'],
    index=['Internal', 'External'])

## Assigning layer information to a dataframe:

# Appending all layer data to a list:
for layer in range(0,n):
    layer_data.append([
        MaterialData.index[ID[layer]],                     # Layer Name
        thck[layer],                                       # Thickness [m]
        MaterialData["k"].iloc[ID[layer]],                 # Linear heat transfer coefficient [W/m/K]
        MaterialData["pi"].iloc[ID[layer]],                # Permeability coefficient  [kg/m/s/Pa]
        thck[layer]/MaterialData["k"].iloc[ID[layer]],     # Thermal Resistance [m²*K/W]
        thck[layer]/MaterialData["pi"].iloc[ID[layer]]     # (Vapour) Resistance [m²*K/W]
        ])

# Converting list to dataframe:

Layer_Props = pd.DataFrame(
    layer_data,
    columns=['LayerName', 'Thickness', 'k', 'Permeability', 'Rth', 'Rv'],
    index=['LAYER {}'.format(i+1) for i in range(0,n)])

print(Layer_Props)

## Total resistances and thermal flow

# Thermal Resistances  
Rth_total = Total_resistance(Conditions.Convection['Internal'], Layer_Props['Rth'], Conditions.Convection['External'], 0)
# Total thermal flow
thermal_flow = Total_flow(Rth_total, Conditions.Temperature['Internal'], Conditions.Temperature['External']) 



###! TEMPERATURE EVOLUTION

# Setting number of ticks and iterations
Magnitude = np.floor(np.log10(thck_tot))    # determine the order of magnitude
ticks = (10**Magnitude)/100;                # assures the graph will always have the correct amount of ticks
# ticks = 0.001;                            # manual choice of ticks
iterations = rounduptoint(thck_tot/ticks)

# Arrays containing layer boundaries
layer_boundaries_real = [round(sum(thck[:i]), 5) for i in range(n+1)]                 # array indicating each layer boundary
layer_boundaries_iters = [rounduptoint(sum(thck[:i])/ticks) for i in range(n+1)]      # layer boundaries in terms of number of iterations
# print(layer_boundaries_real)
# print(layer_boundaries_iters)


## Calculating and storing Temperature Evolution in an array *

# Initial Temperature Value (on inner surface)
Temperature = [Conditions.Temperature['Internal']-(thermal_flow/Conditions.Convection['Internal'])]
# All Temperature Evolution WITHIN THE WALL
for i in range(1, iterations):
    Temperature.append(variable_evolution(Temperature[i-1], thermal_flow, Layer_Props.k[f"LAYER {determine_layer_num(i, layer_boundaries_iters)}"], ticks))
# Final Temperature Value (at outer surface)
Temperature.append(Conditions.Temperature['External']+(thermal_flow/Conditions.Convection['External']))

###! PRESSURE EVOLUTIONS

# Initialising pressure arrays
pvs_vals = [Conditions.SatVapPressure['Internal']]
pv_vals = [Conditions.VapPressure['Internal']]
## Calculating vapour flow through the wall:

# print(Total_resistance(Conditions.MassTransfer['Internal'], Layer_Props['Rv'], Conditions.MassTransfer['External'], 0))
Rv_total = Total_resistance(Conditions.MassTransfer['Internal'], Layer_Props['Rv'], Conditions.MassTransfer['External'], 0)

vapour_flow = Total_flow(Rv_total, pv_vals[0], Conditions.VapPressure['External'])
## Calculating and storing Pressure evolution in arrays
# Filling arrays with a loop
for i in range(1, iterations):
    pvs_vals.append(pvs(Temperature[i]))
    pv_vals.append(variable_evolution(pv_vals[i-1], vapour_flow, Layer_Props.Permeability[f"LAYER {determine_layer_num(i, layer_boundaries_iters)}"], ticks))
# Final Pressure Values
pvs_vals.append(Conditions.SatVapPressure['External'])
pv_vals.append(Conditions.VapPressure['External'])

###! DATAFRAME CONVERSION
x_vals = np.arange(0, round(thck_tot+ticks, 5), ticks); 
x_vals = np.round(x_vals, 5)
Layer_num = [determine_layer_num(x, layer_boundaries_real) for x in x_vals]

Final_df = pd.DataFrame(
    data={'Thickness':x_vals,
          'Temperature': Temperature,
          'SatVapPressure': pvs_vals,
          'VapPressure': pv_vals,
          'Material': [MaterialData.index[ID[i-1]] for i in Layer_num]
          })
Final_df.set_index('Thickness', inplace=True)

# pd.set_option("display.max_rows", None)
# print(Final_df)

###! FINAL CALCULATIONS

## Finding condensation point:
condensation_points = [];
check = True;
for i in range(1, iterations):
    if (np.mean(pv_vals[i-1:i]) >= np.mean(pvs_vals[i-1:i])) and (check==True): # checks if pv>pvs in each iteration
        condensation_points.append(avg(i-1, i)*ticks)        # stores condensation point in the array
        check = False                                       # stops the loop checking once point is located
    if (np.mean(pv_vals[i-1:i]) <= np.mean(pvs_vals[i-1:i])): check = True  # allows loop to start checking after first area of condensation

print ("condenstation point = ", condensation_points)


###! PLOTTING
dfplot(Final_df, layer_boundaries_real)
# plt.show()


###! VAPOUR BARRIER SUGGESTION

VapourBarrierData = pd.read_csv('projectData/VapourBarrierData.csv', index_col=0)
print(VapourBarrierData.columns)

## Checks for condensation
if len(condensation_points) == 0:
    print("It appears there will be no condensation in these conditions")
else:
    print(f"There is some condensation at {condensation_points} m in your wall!")
    print("Here are your choices of Vapour Barriers: \n")
    print(VapourBarrierData)
    VB_ID = int(input("Please input the ID number: "))
    print()
        
    # VB_permeance = VapourBarrierData["Permeance"].iloc[ID[VB_ID]]
    VB_permeance = VapourBarrierData.iloc[VB_ID, 0]
    print(VapourBarrierData.iloc[VB_ID, 0])
    # print(VapourBarrierData["Permeance"].iloc[ID[:,VB_ID]])

    # Rv_1_new = Layer_Props.Rv['LAYER 1']
    # print("Rv_tot = ", Rv_total)
    Rv_total_new = Total_resistance(Conditions.MassTransfer['Internal'], Layer_Props['Rv'], Conditions.MassTransfer['External'], VB_permeance)
    # print("new Rv_tot = ", Rv_total_new)

    # pv_vals_new = [Rv_total_new*vapour_flow+Conditions.SatVapPressure['External']]
    Rv_VB = 1/VB_permeance
    # vapour_flow_VB = Total_flow(Rv_VB, )
    vapour_flow_new = Total_flow(Rv_total_new, Conditions.VapPressure['Internal'], Conditions.VapPressure['External'])
    # print("pv_i = ", Conditions.VapPressure['Internal'])
    # print("vapour flow = ", vapour_flow)
    # print("new vapour flow = ", vapour_flow_new)
    # print(vapour_flow_new*(1/8.83e-12))
    pv_vals_new = [variable_evolution(Conditions.VapPressure['Internal'], vapour_flow_new, VB_permeance, 1)]
    # print(pv_vals_new)

    for i in range(1, iterations):
        pv_vals_new.append(variable_evolution(pv_vals_new[i-1], vapour_flow_new, Layer_Props.Permeability[f"LAYER {determine_layer_num(i, layer_boundaries_iters)}"], ticks))
    pv_vals_new.append(Conditions.VapPressure['External'])

    Final_df_with_VB = Final_df.copy()

    Final_df_with_VB['VapPressure'] = pv_vals_new

    print(Final_df_with_VB)
    dfplot(Final_df_with_VB, layer_boundaries_real)
    plt.show()  


