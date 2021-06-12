#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go

from sklearn.ensemble import  RandomForestClassifier
import sklearn

from sklearn.preprocessing import StandardScaler

import json
import random
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

import math
import time
import pickle

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
#from dash.exceptions import PreventUpdate
from flask import Flask
import os

import base64
import io


import librosa

UPLOAD_DIRECTORY = "assets/"

server = Flask(__name__)
server.secret_key = os.environ.get('secret_key','secret')
app = dash.Dash(name = __name__, server = server,prevent_initial_callbacks=True)


def features_extractor(file, n_mfcc = 30):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc = n_mfcc)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features



@app.callback(Output('save', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              #State('upload-data', 'last_modified')
             )
def parse_contents_audio(contents, filename):
    
    if '.wav' in filename:
    
        data = base64.b64decode(str(contents).split(',')[1])
        fh = open(UPLOAD_DIRECTORY+filename, "wb")
        fh.write(data)
    else:
        return None
    

@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              #State('upload-data', 'last_modified')
             )
def update_output(content, filename):
    

    
    return {None:'',filename:filename}[filename]


@app.callback(Output('output-data-audio', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename')
              
             )
def update_audio(list_of_contents, filename):
    

    
    return html.Audio(src='/assets/'+filename,controls=True, autoPlay = True)


@app.callback(Output('prediction', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename')
             )
def predict(contents,filename):
    
    try:
        features = pd.DataFrame(features_extractor('assets/'+filename)).T

        model = pickle.load(open('assets/model','rb'))
        labelencoder = pickle.load(open('assets/labelencoder','rb'))

        scl = pickle.load(open('assets/scaler','rb'))

        X = scl.transform(features)

        prediction = labelencoder.inverse_transform(model.predict(X))[0].lower()

        return html.P({0:'Oisko {}?'.format(prediction),
                1:'Ehkä {}'.format(prediction),
                2:'Varmaan {}'.format(prediction)}[random.randint(0,2)],style=dict(textAlign='center',
                       fontSize=55, 
                       fontFamily='Arial',
                       color = "blue"))
    except:
        return 'Tiedosto ei ole .wav muotoinen tai on liian iso. Muunna .wav-tiedostoksi tai leikkaa sitä pienemmäksi.'

    



def serve_layout():
    return html.Div([
    html.H1('Oisko Telkkä?', 
            style=dict(textAlign='center',
                       fontSize=55, 
                       fontFamily='Arial',
                       color = "blue")
           ),
    html.Br(),        
    html.P('Tällä sivulla voit ladata nauhoittamasi linnun laulun ja saada arvion mikä lintu on kyseessä. Sivu hyödyntää valmiiksi koulutettua koneoppimismallia, joka on opetettu Xeno-Canto.org-palveluun rekisteröidyillä lintuharrastajien tekemillä äänityksillä. ',style=dict(textAlign='center')),
        html.Br(),
            html.P('Sivulle voi ladata .wav-tyyppisen tiedoston. Ohjelma muodostaa äänitiedostosta Mel-frequency cepstrumin (ks. lisää alla olevasta linkistä), jonka perusteella malli pyrkii tunnistamaan lintulajin.',style=dict(textAlign='center')),
        html.Br(),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Raahaa .wav-tiedosto tähän hiirellä tai ',
            html.A('avaa valikko')
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
        multiple=False
    ),
        html.Div(id='save'),
    html.Div(id='output-data-upload'),
        html.Div(id='output-data-audio'),
        dcc.Loading(children = [html.Div(id = 'prediction')],type ='cube'),

                            html.Label(['Xeno-Canto.org ', 
                                html.A('Lintujen äänitykset Suomessa ', href='https://www.xeno-canto.org/explore?query=area%3A%22europe%22+cnt%3A%22Finland')
                               ]),
                           html.Label(['Mel-frequency Cepstrum ', 
                                html.A('Wikipedia ', href='https://en.wikipedia.org/wiki/Mel-frequency_cepstrum')
                               ]),
                            html.Label(['Get from ', 
                                html.A('GitHub', href='https://github.com/tuopouk/oiskotelkka')
                               ]),
                    html.Label(['by Tuomas Poukkula. ', 
                                html.A('Follow on Twitter.', href='https://twitter.com/TuomasPoukkula')
                               ]),
                    html.Label(['Follow also on ', 
                                html.A('LinkedIn.', href='https://www.linkedin.com/in/tuomaspoukkula/')
                               ])
])




app.title = 'Oisko telkkä?'
app.config['suppress_callback_exceptions']=True  
app.layout = serve_layout

if __name__ == '__main__':
    app.run_server(debug=False)
