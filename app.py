import base64
import datetime
import io 
import cv2
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc 
import dash_html_components as html 
import dash_table
import plotly.graph_objects as go
from TFEmotionClassifier import TFEmotionClassifier
classifier = TFEmotionClassifier()

app = dash.Dash(__name__)
app.layout = html.Div(className='outer-div', children=[
    html.Div(className='content-div', children=[
        dcc.Upload(
            id='upload-image',
            children=html.Div(className='upload-component', children=[
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True
        ),
        html.Div(id='output-image-upload'),
    ])
])

def make_classification(img_base64):
    img_base64 = str.encode(img_base64)
    with open('images/image.png', 'wb') as fh:
        fh.write(base64.decodebytes(img_base64))
    classification, probabilities = classifier.classify('images/image.png')
    return classification, probabilities

def parse_contents(contents, filename, date):
    img = contents.split(',')[-1]
    classification, probabilities = make_classification(img)
    return html.Div([
        html.Img(className='output-img', src=contents),
        html.H3(className='h3-output', children=[classification]),
        dcc.Graph(id='output-chart', figure=go.Figure(
            data=[go.Bar(
                x=probabilities['Probability'],
                y=probabilities['Class'],
                orientation='h'
            )],
            layout=go.Layout()
        ))
    ])


@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


if __name__ == '__main__':
    app.run_server(debug=True)