#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   app_david.py
@Time    :   2022/03/28 14:28:32
@Author  :   Bo 
'''
from ast import Load
import numpy as np 
import os
import h5py 
import dash
from dash import html, dcc
import pandas as pd
import pickle 
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import argparse



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def give_args():
    parser = argparse.ArgumentParser(description='Display SERS maps')
    parser.add_argument("--data_path", type=str, default="../rs_dataset/")
    parser.add_argument("--start_wave", type=int, default=0)
    parser.add_argument("--end_wave", type=int, default=0)
    return parser.parse_args()


def read_mapx(filename):
    """Args:
    filename: the filename that ends with .mapx
    Return:
        spectra: [imh * imw, wavenumber]
        w: wavenumber 
        mapsize: [imh, imw]
    """
    file = h5py.File(filename, 'r')
    N = 0
    info = file['Regions']
    for group in info.keys():
        dataset = info[group]['Dataset']
        N_cur = np.prod(dataset.shape[:2])
        if N_cur > N:
            spectra = dataset[:]
            N = N_cur
    # get mapsize
    mapsize = (spectra.shape[1], spectra.shape[0])
    # extract wavenumbers
    metadata = file['FileInfo'].attrs
    wl_start = metadata['SpectralRangeStart']
    wl_end = metadata['SpectralRangeEnd']
    w = np.linspace(wl_start, wl_end, spectra.shape[2])
    spectra = spectra.T.reshape((spectra.shape[2], np.prod(spectra.shape[:2])), order='C').T
    file.close()
    return spectra, w, mapsize



class LoadData(object):
    def __init__(self, data_path, start_wave, end_wave):
        self.data_path = data_path 
        self.all_sersmaps = self.get_subfiles()
        self.sers_maps = {}
        self.keys = ["filenames", "sers_maps", "wavenumber", "mapsize"]
        for s_k in self.keys:
            self.sers_maps[s_k] = []
        self.start_wave = start_wave
        self.end_wave = end_wave
        
        
    def get_subfiles(self):
        all_files = sorted([v for v in os.listdir(self.data_path) if ".mapx" in v])
        return all_files 
    
    def cut_sersmaps(self, s_map, s_wave):
        if self.start_wave != 0 or self.end_wave != 0:
            start_index = np.where(s_wave >= self.start_wave)[0][0]
            end_index = np.where(s_wave >= self.end_wave)[0][0]
            s_map_update = s_map[:, start_index:end_index]
            s_wave_update = s_wave[start_index:end_index]
        else:
            s_map_update, s_wave_update = s_map, s_wave 
        return s_map_update, s_wave_update
    
    def forward(self):
        for i, s_filename in enumerate(self.all_sersmaps):
            s_map, s_w, s_mapsize = read_mapx(self.data_path + "/" + s_filename)
            s_map, s_w = self.cut_sersmaps(s_map, s_w)
            for s_k, s_value in zip(self.keys, [s_filename.split(".mapx")[0], s_map, s_w, s_mapsize]):
                self.sers_maps[s_k].append(s_value)
        return pd.DataFrame(self.sers_maps)
    
args = give_args()    
data_obj = LoadData(args.data_path, args.start_wave, args.end_wave)
df = data_obj.forward()



app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div(
        dcc.Dropdown(
            id='filename_dropout',
            options=[{'label': "%s" % v, 'value': v} for i, v in enumerate(df.filenames)],
            value=df.filenames.unique()[0]
    ), style={'width': '10%', 'float': 'left', 'display': 'inline-block'}),
    
    html.Div(dcc.Slider(
        id='wavenumber_slider',
        min=df['wavenumber'].iloc[0].min(),
        max=df['wavenumber'].iloc[0].max(),
        value=df['wavenumber'].iloc[0].min(),
        step=1), 
             style={'width': '90%', 'padding': '0px 20px 20px 20px'}),
    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            hoverData={'points': [{'x': 1, 'y': 1, 'z':2 }]}
        )
    ], style={'width': '80%', 'display': 'inline-block', 'padding': '0 0'}),
    html.Div([
        dcc.Graph(id='single_spectrum'),
    ], style={'display': 'inline-block', 'width': '50%', 'padding': '0 0'}),
    
])


@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),
     [dash.dependencies.Input('filename_dropout', 'value'),
      dash.dependencies.Input('wavenumber_slider', 'value')])
def update_graph(filename_index, wavenumber_index):
    select = df.loc[df["filenames"] == filename_index]
    wave = select["wavenumber"].iloc[0]
    select_location = np.where(wave >= wavenumber_index)[0][0]
    sers_map = np.reshape(select["sers_maps"].iloc[0][:, select_location], select["mapsize"].iloc[0])
    
    fig = make_subplots(rows=1, cols=1,  subplot_titles=["sers map"], 
                        horizontal_spacing=0.13)
    fig.add_trace(go.Heatmap(z=sers_map), row=1, col=1, )
    fig.update_layout(title="SERS map (%.2f cm-1)" % (wave[select_location]),
                      title_x=0.5,
                      width=600, height=400, hovermode='closest')
    return fig


@app.callback(
    dash.dependencies.Output('single_spectrum', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
     dash.dependencies.Input('filename_dropout', 'value'),
     dash.dependencies.Input('wavenumber_slider', 'value')])
def update_y_timeseries(hoverData, filename_index, wavenumber_index):
    point_coord = hoverData["points"][0]
    x_loc = point_coord["x"]
    y_loc = point_coord["y"]
    z_int = point_coord["z"]
    select = df.loc[df["filenames"] == filename_index]
    imshape = select["mapsize"].iloc[0]
    spec_index = x_loc + y_loc * imshape[1]
    wave = select["wavenumber"].iloc[0]
    sers_map = select["sers_maps"].iloc[0]
    select_location = np.where(wave >= wavenumber_index)[0][0]
    print("calculate:", sers_map[int(spec_index), select_location], "select", z_int)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wave, y=sers_map[int(spec_index)],
                             mode="lines", name="spectrum"))
    fig.update_xaxes(showgrid=True)
    fig.update_layout(height=400, 
                      title="Spectrum at location (%d, %d)" % (x_loc, y_loc),
                      #title_x=0.5,
                      xaxis_title="Wavenumber(cm-1)",
                      yaxis_title="A.U.",
                      margin=dict(r=400),
                      )
    return fig


if __name__ == "__main__":
    app.run_server(debug=True, port=3003)
