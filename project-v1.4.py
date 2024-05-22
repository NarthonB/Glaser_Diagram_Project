#%%
from datetime import datetime as dt
from meteostat import Daily, Stations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dash import Dash, html, dcc, Input, Output, callback, ALL, Patch, State, dash_table
import plotly.express as px
import plotly.graph_objects as go, plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.io as pio

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


def station_data(ccode, name, start, end): # takes country code, station name, start and end dates
    ccode = str(ccode); name = str(name)

    stations = Stations()
    stations = stations.region(ccode)
    nb_stations = stations.count()
    stationss = stations.fetch(nb_stations)
    for i in range(int(nb_stations)):
        if  stationss["name"].iloc[i] == name : 
            stat_id=stationss.index[i];
        
    if stat_id in stat_id:
        lat = stationss.at[stat_id, 'latitude']
        long = stationss.at[stat_id, 'longitude']
    else: print("Station not Found")
    
    station = stations.nearby(lat,long)
    station = station.fetch(1)
    station_data = Daily(station.index[0], start, end).fetch()
    
    return station_data # returns the data of the found station from the given dates 



## Extracting Layer Data
MaterialData = pd.read_csv('projectData/LayerData.csv')

###! USER INPUT:
## Assigning Properties for each layer:

''' The following variables are not assigned directly into the df
    as they will depend on user input and Meteostat data in a future version. '''

# Internal conditions: (#*# will depend on user's internal requirements)
air_speed_i = 0.2                       #*#
hc_i = convection_coeff(air_speed_i)        # convection coefficient [W/m²K]
# hm_i = 1.11e-3                            # mass transfer coefficient []
hm_i = hc_i*6.9e-9
internal_Temperature = 20.0             #*#
hum_i = 0.5                             #*#
pvs_i = pvs(internal_Temperature)
pv_i =  pvs_i*hum_i

# External conditions: (#*# to depend on méteostat)
air_speed_e = 4.0                       #*#
hc_e = convection_coeff(air_speed_e)    #*#
# hm_e = 1.06E-2
hm_e = hc_e*6.9e-9
external_Temperature = 0.0              #*#
hum_e = 0.9                             #*#
pvs_e = pvs(external_Temperature)
pv_e =  pvs_e*hum_e

# Constructing DataFrame with Internal/External Conditions
Conditions = pd.DataFrame(
    [[hc_i, hm_i, internal_Temperature, hum_i, pvs_i, pv_i],
     [hc_e, hm_e, external_Temperature, hum_e, pvs_e, pv_e]],
    columns=['Convection', 'MassTransfer', 'Temperature', 'RelativeHumidity', 'SatVapPressure', 'VapPressure'],
    index=['Internal', 'External'])

# Country Specs:
# country_code = input("Please enter your country code")
# station_name = input("Please enter your station name")


###! USER INPUT:

layer_data = []
Layer_Property_col_names = ['Material', 'Thickness', 'k', 'Permeability']

import plotly.graph_objects as go, plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.io as pio

default_Materials=['Glass Wool', 'Concrete', 'Brick']
default_Thicknesses=[0.05, 0.25, 0.05]
num_rows_input = dcc.Input(id='num-rows-input', type='number', value=3)

Material_Dropdown_options = [{'label': material, 'value': material} for material in MaterialData['MATERIAL'].unique()]

default_n_rows = 3



app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1('Layer Data Input'),
        html.Div([
            html.Div([
                html.Label('Number of rows:'),
                num_rows_input
        ]),
        html.Button(id='submit-button', children='Submit')
    ]),
        dash_table.DataTable(
            id='editable-table',
            columns=(
            [{'id': 'Layer', 'name': 'Layer'}] +
            [{'id': p, 'name': p, 'presentation':
                {'type': 'input'}} if p != 'Material' 
                else {'id': p, 'name': p, 'presentation': 'dropdown'} for p in Layer_Property_col_names]
            ),
                # {'id': 'Layer', 'name': 'Layer Number'},
                # {'id': 'Material', 'name': 'Material', 'dropdown': True, 'options': [{'label': material, 'value': material} for material in MaterialData['MATERIAL'].unique()]},
                # {'id': 'Thickness', 'name': 'Thickness'},
                # {'id': 'k', 'name': 'k'},
                # {'id': 'Permeability', 'name': 'Permeability'}
            # ],

            data=[
                dict(Layer=i,
                    Model=i, 
                     Material=default_Materials[i-1], 
                     Thickness=default_Thicknesses[i-1])
                for i in range(1, default_n_rows+1)                
            ],
            editable=True,
            dropdown={
                'Material': {
                    'options': [{'label': material, 'value': material} for material in MaterialData['MATERIAL'].unique()]
                }
            },
            row_deletable=True            
        ),
    ]),
    # html.Div([
    html.Button('Add Row', id='add-row-button', n_clicks=0),
    # html.Button('Remove Row', id='remove-row-button', n_clicks=0)
    # ]),            
    html.Div(id='editable-table-output'),
    dcc.Store(id='columns-state', data=Layer_Property_col_names),
    html.Button('Calculate', id='calculate-button', n_clicks=1),
    # dash_table.DataTable(
    #     id='Final-DataFrame',
    #     columns=[
    #         {"name": "Thickness [m]", "id": 'Thickness'},
    #         {"name": "Temperature [℃]", "id": 'Temperature'},
    #         {"name": "Saturated Vapour Pressure [Pa]", "id": 'SatVapPressure'},
    #         {"name": "Vapour Pressure [Pa]", "id": 'VapPressure'}
    #     ]   
    # ),
    dcc.Graph(id='Final-Graph'),
    dcc.Store(id='final-data-store'),
    dcc.Store(id='layer-boundaries-store')
])

import dash

@app.callback(
    Output('table-data', 'data'),
    Input('editable-table', 'dropdown_value'))
def update_row(dropdown_value):
    # Find the row with the selected material
    row = MaterialData[MaterialData['MATERIAL'] == dropdown_value].iloc[0]
    # Create a new row with the updated values
    new_row = dict(Model=row['Model'],
                   Material=row['MATERIAL'],
                   k=row['k'],
                   Permeability=row['pi'],
                   Thickness=float(row['Thickness'])
                   )
    # Find the index of the row to update
    data = dash.callback_context.states['editable-table.data']
    index = next((i for i, r in enumerate(data) if r['Model'] == new_row['Model']), None)
    # Update the row if it exists, else append it
    if index is not None: data[index] = new_row
    else: data.append(new_row)
    return data

@app.callback(
    Output('editable-table', 'data'),
    Input('editable-table', 'data'),
    State('columns-state', 'data'))
def update_k_permeability(data, columns):
    # Update the 'k' and 'Permeability' columns based on the selected material
    for row in data:
        material = row['Material']
        if material:
            row['k'] = MaterialData[MaterialData['MATERIAL'] == material]['k'].values[0]
            row['Permeability'] = MaterialData[MaterialData['MATERIAL'] == material]['pi'].values[0]
    return data


# @app.callback(
#     Output('editable-table', 'data'),
#     Input('add-row-button', 'n_clicks'),
#     State('editable-table', 'data'),
#     State('editable-table', 'columns'))
# def add_row(n_clicks, rows, columns):
#     if n_clicks>0:
#         rows.append({c['id']: '' for c in columns})
#     return rows


@app.callback(
    Output('editable-table-output', 'figure'),
    Input('editable-table', 'data'),
    Input('editable-table', 'columns'),
    State('data-store', 'data'))
def display_output(data, columns):
    df = pd.DataFrame(data, columns=[c['name'] for c in columns])
    return {
        'data': [{
            'type': 'parcoords',
            'dimensions': [{
                'label': col['name'],
                'values': df[col['id']]
            } for col in columns]
        }]
    }
   

import plotly.graph_objects as go
 
 
@app.callback(
    [Output('final-data-store', 'data'), Output('layer-boundaries-store', 'data')],
    Input('calculate-button', 'n_clicks'),
    State('editable-table', 'data'),
    State('editable-table', 'columns')
)
def calcs(n_clicks, data, columns):
    # Create a pandas DataFrame from the data and columns inputs
    df = pd.DataFrame(data, columns=[c['name'] for c in columns])
    df['Thickness'] = pd.to_numeric(df['Thickness'])
    
    thickness = df[columns[2]['id']]
    k = df[columns[3]['id']]
    pi = df[columns[4]['id']]
    
    ###! CONSTANT VALUES
    
    total_thickness = sum(thickness)
    # Layer Resistances   
    R_thermal = [t/k for t, k in zip(thickness, k)]
    R_vapour = [t/pi for t, pi in zip(thickness, pi)]
    # Total Thermal Resistance
    R_thermal_total = Total_resistance(Conditions.Convection['Internal'], R_thermal, Conditions.Convection['External'], 0)
    # Total Thermal Flow
    thermal_flow = Total_flow(R_thermal_total, Conditions.Temperature['Internal'], Conditions.Temperature['External']) 
    
    # Setting number of ticks and iterations
    Magnitude = np.floor(np.log10(total_thickness))    # determine the order of magnitude
    ticks = (10**Magnitude)/100;                       # assures the graph will always have the correct amount of ticks
    # ticks = 0.001;                                   # manual choice of ticks

    layer_boundaries_real = [round(sum(thickness[:i]), 5) for i in range(4)]                 # array indicating each layer boundary
    layer_boundaries_iters = [rounduptoint(sum(thickness[:i])/ticks) for i in range(4)]      # layer boundaries in terms of number of iterations
    
    ###! TEMPERATURE EVOLUTION
    
    iterations = rounduptoint(total_thickness/ticks)
    Temperature = [Conditions.Temperature['Internal']-(thermal_flow/Conditions.Convection['Internal'])]
    # All Temperature Evolution WITHIN THE WALL
    for i in range(1, iterations):
        Temperature.append(variable_evolution(Temperature[i-1], thermal_flow, k[determine_layer_num(i, layer_boundaries_iters)-1], ticks))
    # Final Temperature Value (at outer surface)
    Temperature.append(Conditions.Temperature['External']+(thermal_flow/Conditions.Convection['External']))
    ###! PRESSURE EVOLUTIONS

    # Initialising pressure arrays
    pvs_vals = [Conditions.SatVapPressure['Internal']]
    pv_vals = [Conditions.VapPressure['Internal']]

    ## Calculating vapour flow through the wall:
    # Total Vapour Resistance
    R_vapour_total = Total_resistance(Conditions.MassTransfer['Internal'], R_vapour, Conditions.MassTransfer['External'], 0)
    # Total Vapour Flow 
    vapour_flow = Total_flow(R_vapour_total, pv_vals[0], Conditions.VapPressure['External'])

    ## Calculating and storing Pressure evolution in arrays
    # Filling arrays with a loop
    for i in range(1, iterations):
        pvs_vals.append(pvs(Temperature[i]))
        pv_vals.append(variable_evolution(pv_vals[i-1], vapour_flow, pi[determine_layer_num(i, layer_boundaries_iters)-1], ticks))
    # Final Pressure Values
    pvs_vals.append(Conditions.SatVapPressure['External'])
    pv_vals.append(Conditions.VapPressure['External'])
    
    ###! DATAFRAME CONVERSION
    x_vals = np.arange(0, round(total_thickness+ticks, 5), ticks); 
    x_vals = np.round(x_vals, 5)
    Layer_num = [determine_layer_num(x, layer_boundaries_real) for x in x_vals]

    Final_df = pd.DataFrame(
        data={'Thickness':x_vals,
            'Temperature': Temperature,
            'SatVapPressure': pvs_vals,
            'VapPressure': pv_vals,
            'Layer Num': [MaterialData.index[i-1] for i in Layer_num]
            })
    # Final_df.set_index('Thickness', inplace=True)
    # Final_df.reset_index(inplace=True)
    
    return Final_df.to_dict('records'), layer_boundaries_real

@app.callback(
    Output('Final-Graph', 'figure'),
    [Input('final-data-store', 'data'),
     Input('layer-boundaries-store', 'data'),
     Input('editable-table', 'data')]
)
def update_graph(data, layer_boundaries_real, table_data):
    if data is not None:
        ###! Extracting Input Data
        Final_df = pd.DataFrame.from_records(data)
        Final_df.set_index('Thickness', inplace=True)
        
        ## Adding condensation values to the dataframe 
        Final_df['VapPressure_cond'] = np.where(Final_df['VapPressure'] > Final_df['SatVapPressure'], Final_df['VapPressure'], np.nan)
        Final_df['SatVapPressure_cond'] = np.where(Final_df['VapPressure'] > Final_df['SatVapPressure'], Final_df['SatVapPressure'], np.nan)
            
        
        ###! Creating Traces
        # Creating secondary axis:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Temperature Trace:
        Temperature = go.Scatter(x=Final_df.index, y=Final_df['Temperature'],
                                 line=dict(color='red', dash='dash'), name='Temperature')
        # Vapour Pressure trace:
        Vap_Pressure = go.Scatter(x=Final_df.index, y=Final_df['VapPressure'],
                                line=dict(color='cyan'),name='p_v')
        # Create the SatVap_Pressure trace
        SatVap_Pressure = go.Scatter(x=Final_df.index, y=Final_df['SatVapPressure'],
                                    line=dict(color='blue'), name='p_sat')

        ## Creating Condensation Traces
        Vap_Cond = go.Scatter(x=Final_df.index, y=Final_df['VapPressure_cond'],
                                   line=dict(color='cyan'),
                                   fill='tonexty', connectgaps=False)

        SatVap_Cond = go.Scatter(x=Final_df.index, y=Final_df['SatVapPressure_cond'],
                                    line=dict(color='blue'), connectgaps=False)

        if Temperature.y[0] is not None:
            y_min, y_max = min(Temperature.y), max(Temperature.y) * 1.05
        else: y_min, y_max = 0, 25
        
        fig.update_yaxes(range=[y_min, y_max], title_text="Temperature [°C]", showgrid=False, secondary_y=False)
        fig.update_yaxes(title_text="Pressure [Pa]", showgrid=False, secondary_y=True)
        fig.update_xaxes(title_text="distance through wall [m]")
        
        fig.add_trace(Temperature, secondary_y=False)
        fig.add_trace(SatVap_Pressure, secondary_y=True)
        
        ## Displaying Condensation Area 
        fig.add_trace(SatVap_Cond, secondary_y=True)
        fig.add_trace(Vap_Cond, secondary_y=True)
        
        fig.add_trace(Vap_Pressure, secondary_y=True)
        # fig.add_trace(Condensation_Pressure, secondary_y=True)
        
        fig.update_xaxes(showgrid=False)
        fig.update_layout(showlegend=False)
        
        # Update the color of each layer based on the material
        material_colours = [colourpicker(row['Material']) for row in table_data]

        ## Displaying layer Boundaries
        Final_df['Color'] = Final_df['Layer Num'].apply(lambda x: material_colours[x])

        shapes = []
        for i, boundary in enumerate(layer_boundaries_real):
            fillcolour = material_colours[i] if i < len(material_colours) else material_colours[-1]
            shape = {
                'type': 'rect',
                'x0': boundary,
                'x1': layer_boundaries_real[i+1] if i+1 < len(layer_boundaries_real) else max(layer_boundaries_real),
                'y0': 0,
                'y1': y_max,
                'fillcolor': fillcolour,
                'opacity': 0.5,
                'layer': 'below',
                'line': {'width': 0}
            }
            shapes.append(shape)

        # updates colours if material is changed
        fig.update_layout(shapes=shapes)
        
        layer_boundaries_array = [(boundary, boundary) for boundary in layer_boundaries_real]
        for boundary in layer_boundaries_array:
            fig.add_shape(type='line', x0=boundary[0], y0=0, x1=boundary[1], y1=100, line=dict(color='black', width=2))
        
        # Shade Condensation Area :
        # Cond_p_vs, Cond_p_v = []
        # # for i, j in Final_df['VapPressure'], Final_df['SatVapPressure']:
        # for i in range(Final_df['VapPressure']):
        #     if Final_df["VapPressure"]>Final_df["SatVapPressure"]:
            

        return fig
    else: return go.Figure()
    
if __name__ == '__main__':
    app.run()

#%%
###! CALCULATION OF CONSTANT VALUES


# ## Total resistances and thermal flow

# # Thermal Resistances  
# Rth_total = Total_resistance(Conditions.Convection['Internal'], Layer_Props['Rth'], Conditions.Convection['External'], 0)
# # Total thermal flow
# thermal_flow = Total_flow(Rth_total, Conditions.Temperature['Internal'], Conditions.Temperature['External']) 



# ###! TEMPERATURE EVOLUTION

# # Setting number of ticks and iterations
# Magnitude = np.floor(np.log10(thck_tot))    # determine the order of magnitude
# ticks = (10**Magnitude)/100;                # assures the graph will always have the correct amount of ticks
# # ticks = 0.001;                            # manual choice of ticks
# iterations = rounduptoint(thck_tot/ticks)

# # Arrays containing layer boundaries
# layer_boundaries_real = [round(sum(thck[:i]), 5) for i in range(n+1)]                 # array indicating each layer boundary
# layer_boundaries_iters = [rounduptoint(sum(thck[:i])/ticks) for i in range(n+1)]      # layer boundaries in terms of number of iterations
# # print(layer_boundaries_real)
# # print(layer_boundaries_iters)


# ## Calculating and storing Temperature Evolution in an array *

# # Initial Temperature Value (on inner surface)
# Temperature = [Conditions.Temperature['Internal']-(thermal_flow/Conditions.Convection['Internal'])]
# # All Temperature Evolution WITHIN THE WALL
# for i in range(1, iterations):
#     Temperature.append(variable_evolution(Temperature[i-1], thermal_flow, Layer_Props.k[f"LAYER {determine_layer_num(i, layer_boundaries_iters)}"], ticks))
# # Final Temperature Value (at outer surface)
# Temperature.append(Conditions.Temperature['External']+(thermal_flow/Conditions.Convection['External']))

# ###! PRESSURE EVOLUTIONS

# # Initialising pressure arrays
# pvs_vals = [Conditions.SatVapPressure['Internal']]
# pv_vals = [Conditions.VapPressure['Internal']]
# ## Calculating vapour flow through the wall:

# # print(Total_resistance(Conditions.MassTransfer['Internal'], Layer_Props['Rv'], Conditions.MassTransfer['External'], 0))
# Rv_total = Total_resistance(Conditions.MassTransfer['Internal'], Layer_Props['Rv'], Conditions.MassTransfer['External'], 0)

# vapour_flow = Total_flow(Rv_total, pv_vals[0], Conditions.VapPressure['External'])
# ## Calculating and storing Pressure evolution in arrays
# # Filling arrays with a loop
# for i in range(1, iterations):
#     pvs_vals.append(pvs(Temperature[i]))
#     pv_vals.append(variable_evolution(pv_vals[i-1], vapour_flow, Layer_Props.Permeability[f"LAYER {determine_layer_num(i, layer_boundaries_iters)}"], ticks))
# # Final Pressure Values
# pvs_vals.append(Conditions.SatVapPressure['External'])
# pv_vals.append(Conditions.VapPressure['External'])

# ###! DATAFRAME CONVERSION
# x_vals = np.arange(0, round(thck_tot+ticks, 5), ticks); 
# x_vals = np.round(x_vals, 5)
# Layer_num = [determine_layer_num(x, layer_boundaries_real) for x in x_vals]

# Final_df = pd.DataFrame(
#     data={'Thickness':x_vals,
#           'Temperature': Temperature,
#           'SatVapPressure': pvs_vals,
#           'VapPressure': pv_vals,
#           'Material': [MaterialData.index[ID[i-1]] for i in Layer_num]
#           })
# Final_df.set_index('Thickness', inplace=True)

# # Final_df.to_csv('3layerexample.csv', index=False)

# # pd.set_option("display.max_rows", None)
# # print(Final_df)

# ###! FINAL CALCULATIONS

# ## Finding condensation point:
# condensation_points = [];
# check = True;
# for i in range(1, iterations):
#     if (np.mean(pv_vals[i-1:i]) >= np.mean(pvs_vals[i-1:i])) and (check==True): # checks if pv>pvs in each iteration
#         condensation_points.append(avg(i-1, i)*ticks)        # stores condensation point in the array
#         check = False                                       # stops the loop checking once point is located
#     if (np.mean(pv_vals[i-1:i]) <= np.mean(pvs_vals[i-1:i])): check = True  # allows loop to start checking after first area of condensation

# print ("condenstation point = ", condensation_points)


# ###! PLOTTING
# dfplot(Final_df, layer_boundaries_real)
# # plt.show()


# ###! VAPOUR BARRIER SUGGESTION

# VapourBarrierData = pd.read_csv('projectData/VapourBarrierData.csv', index_col=0)
# print(VapourBarrierData.columns)

# ## Checks for condensation
# if len(condensation_points) == 0:
#     print("It appears there will be no condensation in these conditions")
# else:
#     print(f"There is some condensation at {condensation_points} m in your wall!")
#     print("Here are your choices of Vapour Barriers: \n")
#     print(VapourBarrierData)
#     VB_ID = int(input("Please input the ID number: "))
#     print()
        
#     # VB_permeance = VapourBarrierData["Permeance"].iloc[ID[VB_ID]]
#     VB_permeance = VapourBarrierData.iloc[VB_ID, 0]
#     print(VapourBarrierData.iloc[VB_ID, 0])
#     # print(VapourBarrierData["Permeance"].iloc[ID[:,VB_ID]])

#     # Rv_1_new = Layer_Props.Rv['LAYER 1']
#     # print("Rv_tot = ", Rv_total)
#     Rv_total_new = Total_resistance(Conditions.MassTransfer['Internal'], Layer_Props['Rv'], Conditions.MassTransfer['External'], VB_permeance)
#     # print("new Rv_tot = ", Rv_total_new)

#     # pv_vals_new = [Rv_total_new*vapour_flow+Conditions.SatVapPressure['External']]
#     Rv_VB = 1/VB_permeance
#     # vapour_flow_VB = Total_flow(Rv_VB, )
#     vapour_flow_new = Total_flow(Rv_total_new, Conditions.VapPressure['Internal'], Conditions.VapPressure['External'])
#     # print("pv_i = ", Conditions.VapPressure['Internal'])
#     # print("vapour flow = ", vapour_flow)
#     # print("new vapour flow = ", vapour_flow_new)
#     # print(vapour_flow_new*(1/8.83e-12))
#     pv_vals_new = [variable_evolution(Conditions.VapPressure['Internal'], vapour_flow_new, VB_permeance, 1)]
#     # print(pv_vals_new)

#     for i in range(1, iterations):
#         pv_vals_new.append(variable_evolution(pv_vals_new[i-1], vapour_flow_new, Layer_Props.Permeability[f"LAYER {determine_layer_num(i, layer_boundaries_iters)}"], ticks))
#     pv_vals_new.append(Conditions.VapPressure['External'])

#     Final_df_with_VB = Final_df.copy()

#     Final_df_with_VB['VapPressure'] = pv_vals_new

#     print(Final_df_with_VB)
#     dfplot(Final_df_with_VB, layer_boundaries_real)
#     plt.show()  


