import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LayerData = pd.read_csv('projectData/LayerData.csv', index_col=0)

### USER INPUT:
n = int(input("Please enter the number of layers: ")) # number of layers
print()
from tabulate import tabulate

layer_data = []
thck = [0]*n; ID = [0]*n
k = [0]*n
pi = [0]*n
R = [0]*n
Rv = [0]*n


for i in range(n):
    print("-" * 50)
    print("Let's get data for layer", i+1,":")
    print("Here is the choice of materials: \n")
    print(LayerData); print()
    ID[i] = int(input(f"Insert the ID number of your material for layer {i + 1}: "))
    thck[i] = float(input(f"Layer thickness [m] for layer {i + 1}: "))
        
    print("-" * 50, end="\n")


if n>0: thck_1 = thck[0]; ID_1 = ID[0]
if n>1: thck_2 = thck[1]; ID_2 = ID[1]
if n>2: thck_3 = thck[2]; ID_3 = ID[2]
if n>3: thck_4 = thck[3]; ID_3 = ID[3]

#!# to be put into an array ^

### immediate calcs:
thck_tot = sum(thck)

#######

# establish data in each layer:
## Int: hc, hm, RH, T_i,;
hc_i = 1/0.14
hm_i = 1.11E-3
T_i = 20 #!# will be based on internal requirements (user input?)
hum_i = 0.5

# Layer 1:
if n > 0:
    k_1 = LayerData["k"].iloc[ID[0]]
    pi_1 = LayerData["pi"].iloc[ID[0]]
    R_1 = thck[0]/k_1
    Rv_1 = thck[0]/pi_1

## Layer 2:
if n > 1:
    k_2 = LayerData["k"].iloc[ID[1]]
    pi_2 = LayerData["pi"].iloc[ID[1]]
    R_2 = thck[1]/k_2
    Rv_2 = thck[1]/pi_2

## Layer 3: 
if n > 2:
    k_3 = LayerData["k"].iloc[ID[2]]
    pi_3 = LayerData["pi"].iloc[ID[2]]
    R_3 = thck[2]/k_3
    Rv_3 = thck[2]/pi_3
    
#!# to be automated ^

## Ext:
hc_e = 1/0.04
hm_e = 1.06E-2
    #!# following will depend on méteostat:
T_e = 0
hum_e = 0.9

# summary array:
if n==1: Layr_th = [R_1]; Layr_v = [Rv_1]
if n==2: Layr_th = [R_1, R_2]; Layr_v = [Rv_1, Rv_2]
if n==3: Layr_th = [R_1, R_2, R_3]; Layr_v = [Rv_1, Rv_2, Rv_3]

#!# to be automated ^

#Heat and Moisture Transfer
Rth_wall,Rv_wall=0,0
for i in range(n):
    Rth_wall += Layr_th[i]
    Rv_wall += Layr_v[i]

Rth_tot = (1/hc_i)+Rth_wall+(1/hc_e)
flux_c=(1/Rth_tot)*(T_i-T_e)

Rv_total = (1/hm_i)+Rv_wall+(1/hm_e)


# Temperature Evolution

ticks = 0.0001;
itrs = thck_tot/ticks;
if n>0: x_1 = thck[0]/ticks
if n>1: x_2 = (thck[0]+thck[1])/ticks
if n>2: x_3 = (thck[0]+thck[1]+thck[2])/ticks

T = [T_i-(flux_c/hc_i)]

for i in range(1, int(itrs)):
    if (n>0) and (i <= x_1): T.append(T[i-1]-(flux_c*(ticks/k_1))) #!#
    elif (n>1) and (x_1 < i) and (i <= x_2): T.append(T[i-1]-(flux_c*(ticks/k_2))) #!#
    elif (n>2) and (x_2 < i) and (i <= x_3): T.append(T[i-1]-(flux_c*(ticks/k_3))) #!#


  # Pressures
def pvs(Temp): pvs = np.power(10, (2.7877+7.625*(Temp/(241.6+Temp)))); return pvs

pvs_vals = [pvs(T[0])]
pv_vals = [hum_i*pvs_vals[0]]

pvs_e = pvs(T_e)
pv_e = hum_e*pvs_e
flux_m=(1/Rv_total)*(pv_vals[0]-pv_e)

for i in range(1, int(itrs)):
    pvs_vals.append(pvs(T[i]))
    
    if (n>0) and (i <= x_1): pv_vals.append(pv_vals[i-1]-(flux_m*(ticks/pi_1))) #!#
    elif (n>1) and (x_1 < i) and (i <= x_2): pv_vals.append(pv_vals[i-1]-(flux_m*(ticks/pi_2))) #!#
    elif (n>2) and (x_2 < i) and (i <= x_3): pv_vals.append(pv_vals[i-1]-(flux_m*(ticks/pi_3))) #!#
    
### LIST CONVERSION
# import array as arr
T = np.array(T)
pv_vals = np.array(pv_vals)
pvs_vals = np.array(pvs_vals)
x_vals = np.arange(ticks, thck_tot, ticks); x_vals = np.array(x_vals)

### PLOTTING

fig, ax1 = plt.subplots()

ax1.plot(x_vals, T, '#F62817', label='Temperature [°C]', linewidth=0.85)
ax1.set_xlabel('Position [m]')
ax1.set_ylabel('Temperature [°C]', color='k')
ax1.tick_params('y', colors='k')

if n>1: ax1.axvline(x=x_1*ticks, color='k', linestyle='-', linewidth=0.95) #!#
if n>2: ax1.axvline(x=x_2*ticks, color='k', linestyle='-', linewidth=0.95) #!#

ax2 = ax1.twinx()
ax2.plot(x_vals, pvs_vals, 'b', linestyle='--', label='pvs', linewidth=0.95)
ax2.set_ylim(0, (max(pv_vals)+400))
ax2.set_ylabel('pvs', color='k')
ax2.tick_params('y', colors='k')

ax2.plot(x_vals, pv_vals, 'cyan', linestyle='--', label='pv', linewidth=0.95)

ax2.fill_between(x_vals, pv_vals, pvs_vals, where=(pv_vals >= pvs_vals), color='blue', alpha=0.2)

ax1.legend(loc='upper center')
ax2.legend(loc='upper right')

plt.show()



