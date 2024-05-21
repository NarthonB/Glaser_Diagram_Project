
'''This file is being used to prepare a dash interface for the existing code'''

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.


import dash
from dash import Dash, html, dcc, Input, Output, callback, State
# import plotly.express as px
import numpy as np, pandas as pd
# import plotly.graph_objects as go, plotly.subplots as sp
# from plotly.subplots import make_subplots
# import plotly.io as pio

# Layer_Data = pd.read_csv('projectData/LayerData.csv')
MaterialData = pd.read_csv('projectData/LayerData.csv')

MAX_LAYERS = 10
app = Dash(__name__)

app.layout = html.Div([
    html.H1('Layer Data Input'),
    # html.P('Enter data for each layer:'),
    html.Div(id='layer-inputs'),
    html.Button(
        id='add-layer',
        children='Add Layer',
        n_clicks=0
    ),
    html.Div(id='output')
])

@app.callback(
    Output('layer-inputs', 'children'),
    [Input('add-layer', 'n_clicks')],
    [State('layer-inputs', 'children')]
)
def generate_layer_inputs(n_clicks, children):
    if children is None:
        inputs = []
    else: inputs = children
    
    if n_clicks > 0:
        last_index = len(inputs)
        inputs.append(html.Div([
            html.H2(f'Layer {last_index+1}:'),
            html.P('Select material:'),     # should include the relevant data on hover
            dcc.Dropdown(
                id=f'material-{last_index}',
                options=[{'label': material, 'value': material} for material in MaterialData['MATERIAL'].unique()]
            ),
            html.P('Enter layer thickness [m]:'),
            dcc.Input(
                id=f'thickness-{last_index}',
                type='number',
                value=0,
            )
        ]))
    return inputs

@app.callback(
    Output('output', 'children'),
    [Input(f'material-{i}', 'value') for i in range(MAX_LAYERS)] +
    [Input(f'thickness-{i}', 'value') for i in range(MAX_LAYERS)]
)
def process_layer_data(*args):
    ID = []
    thck = []
    for i in range(len(args)//2):
        ID.append(args[i])
        thck.append(args[i + len(args)//2])
    # do something with ID and thck
    return 'Data processed!'


if __name__ == '__main__':
    app.run_server()





# Final_df = pd.read_csv('3layerexample.csv')
# # df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')

# app = Dash(__name__)

# app.layout = html.Div([
#         html.Div([
#                 'Layer 1' 
#             ]),
#             # 'dynamic' so that layers keep appearing until user doesn't need any more
#             # will require a previous question asking for number of layers
#             dcc.Dropdown(
#                 Layer_Data['MATERIAL'].unique(),
#                 # 'Material',
#                 id='Material',
#                 value='Material',
#                 placeholder="Select Material"
#             ),
#             dcc.Slider(
#                 0,
#                 1,
#                 step=None,
#                 value=0,
#                 id='thickness'
#             ),
#             # dcc.Markdown(children='', id='LayerProps'),
#             dcc.Graph(id='OutputGraph'),
#             # dcc.Input(id='thickness', value='', type='text'),

#             html.Br(),
#             html.Div(id='myoutput'),
#         ])     
    

    
# @app.callback(
#     Output(component_id='OutputGraph', component_property='figure'),
#     # Output(component_id='LayerProps', component_property='children'),
#     Input(component_id='Material', component_property='value'),
#     Input(component_id='thickness', component_property='value')
# )
# def Layer_Design(Material, thickness):
#     k = Layer_Data['k'].loc[Material]
#     pi = Layer_Data['pi'].loc[Material]
    
#     fig = make_subplots(specs=[[{"secondary_y": True}]])

#     Temp = go.Scatter(x=Final_df.index, y=Final_df['Temperature'], name='Temperature')
#     Sat_Vap_Pressure = go.Scatter(x=Final_df.index,y=Final_df['SatVapPressure'], name='P_vs')
#     Vap_Pressure = go.Scatter(x=Final_df.index, y=Final_df['VapPressure'], name='p_v')

#     # fig = go.Figure()
#     fig.add_trace(Temp, secondary_y=False)
#     fig.add_trace(Sat_Vap_Pressure, secondary_y=True)
#     fig.add_trace(Vap_Pressure, secondary_y=True)

#     fig.update_yaxes(title_text="Temperature", secondary_y=False)
#     fig.update_yaxes(title_text="Pressure", secondary_y=True)

#     return fig
    
#     # x=np.arange(0, thickness, 0.01); 
#     # # x = np.array(x)
#     # y = [i*2 for i in x]
#     # y = np.array(y)
#     # fig = px.line(y)
#     # return fig
    
    
    
    
# #     Output('outputgraph', 'children'),
# #     [Input('Material1', 'value'),
# #      Input('Material2', 'value'),
# #      Input('Material3', 'value')])
# # def Layer_Props(input1, input2, ):

# #     return dcc.Graph()



# # Running the app
# if __name__ == '__main__':
#     app.run(debug=True)