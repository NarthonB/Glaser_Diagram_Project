#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

### USER INPUT:

n = 2; # number of layers
e_1 = 0.1
e_2 = 0.1

### immediate calcs:
e_tot = e_1+e_2 #!#



#######

# establish data in each layer:
## Int: hc, hm, RH, T_i,;
hc_i = 1/0.14
hm_i = 1.11E-3
    # following based on internal requirements
T_i = 20
hum_i = 0.5

## Layer 1: Laine de Verre (to be translated)
LayerData = pd.read_csv('projectData/LayerData.csv', index_col=0)

k_1 = LayerData["k"].iloc[0]
print(k_1)
pi_1 = LayerData["pi"].iloc[0]
R_1 = e_1/k_1
Rv_1 = e_1/pi_1

## Layer 2: Concrete
k_2 = LayerData["k"].iloc[1]
pi_2 = LayerData["pi"].iloc[1]
R_2 = e_2/k_2
Rv_2 = e_2/pi_2

## Ext:
hc_e = 1/0.04
hm_e = 1.06E-2
    # following from méteostat:
T_e = 0
hum_e = 0.9

# summary array:
Layr_th = [R_1, R_2]
Layr_v = [Rv_1, Rv_2]


#Heat and Moisture Transfer
Rth_wall,Rv_wall=0,0
for i in range(n):
    Rth_wall += Layr_th[i]
    Rv_wall += Layr_v[i]

Rth_tot = (1/hc_i)+Rth_wall+(1/hc_e)
print("Rth_tot = ", Rth_tot)
flux_c=(1/Rth_tot)*(T_i-T_e)
print("flux = ", flux_c)
    
Rv_total = (1/hm_i)+Rv_wall+(1/hm_e)


# Temperature Evolution
T_1=T_i-(flux_c/hc_i)
print("T_1 = ", T_1)
T_2=T_1-(flux_c*R_1) #!#
print("T_2 = ", T_2)
T_3=T_2-(flux_c*R_2) #!#
print("T_3 = ", T_3)

# T = []
# T[0] = T_i
ticks = 0.001;
itrs = e_tot/ticks;
T = [T_i-(flux_c/hc_i)]
T[0] = T_i-(flux_c/hc_i)

for i in range(1, int(itrs)):
    if i<(e_1/ticks): T.append(T[i-1]-(flux_c*(ticks/k_1))) #!#
    else: T.append(T[i-1]-(flux_c*(ticks/k_2))) #!#

x_vals = np.arange(0,e_tot, ticks)


# Pressures
def pvs(Temp): pvs = np.power(10, (2.7877+7.625*(Temp/(241.6+Temp)))); return pvs



pvs_vals = [pvs(T[0])]
pv_vals = [hum_i*pvs_vals[0]]

pvs_e = pvs(T_e)
pv_e = hum_e*pvs_e
flux_m=(1/Rv_total)*(pv_vals[0]-pv_e)

for i in range(1, int(itrs)):
    pvs_vals.append(pvs(T[i]))
    
    if i<(e_1/ticks): pv_vals.append(pv_vals[i-1]-(flux_m*(ticks/pi_1)))
    else: pv_vals.append(pv_vals[i-1]-(flux_m*(ticks/pi_2)))
    
    
plt.plot(x_vals, T)
plt.plot(x_vals, pvs_vals)
plt.plot(x_vals, pv_vals)


