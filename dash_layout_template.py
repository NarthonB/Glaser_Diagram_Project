
'''This file is being used to prepare a dash interface for the existing code'''

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.


from dash import Dash, html, dcc, Input, Output, callback
import plotly.express as px

import pandas as pd

Layer_Data = pd.read_csv('projectData/LayerData.csv')

app = Dash(__name__)

app.layout = html.Div(
    html.Div(
        html.Div([
            dcc.Dropdown(
                ['Layer 1', 'Layer 3', 'Layer 3'],
                'Layer 1',
            ),
            dcc.RadioItems(
                Layer_Data['MATERIAL'].unique(),
                'Material',
                id='Material',
                inline=True
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),
    )
)

if __name__ == '__main__':
    app.run(debug=True)