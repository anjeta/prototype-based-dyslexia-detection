# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:46:40 2024

@author: aneta.kartali
"""

import altair as alt
import base64
import glob
from io import BytesIO
import json
import math
import matplotlib.animation as animation
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import scipy.io
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import sys
import time as timeLib
import torch
import torch.nn as nn

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from features.build_features import feature_extraction, fixation_detection
from data.params.data_params_Benfatto import data_params

# -----------------------------------------------------------------------------
# PAGE SETUP
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Eye-tracking analysis",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# -----------------------------------------------------------------------------
# LEFT SIDEBAR
# -----------------------------------------------------------------------------
class_names = ["HEALTHY", "DYSLEXIA"] 
class_mapping = {'bp': 0, 'dys': 1}
split_char = '\\'

norm_data = {}
data_path = os.path.join(data_params.data_path)
dir_path = os.path.realpath(data_path)
search_path = os.path.join(dir_path, data_params.segmented_dataset, "*\*\*\*." + data_params.norm_ext)

for fname in glob.iglob(search_path, recursive=True):
    subject = fname.split(split_char)[-3]
    with open(fname, "rb") as f:
        norm_data[subject] = pickle.load(f)

with st.sidebar:
    st.title('👁️ Eye-tracking data selection')
    uploaded_file = st.file_uploader("Choose a .json file containing eye-tracking data", accept_multiple_files=False)
    
    if uploaded_file is not None:
        
        file_data = json.loads(uploaded_file.getvalue().decode('utf-8'))
        
        subject = file_data['subject']
        diagnosis = file_data['diagnosis']
        task = file_data['task']
        segment = file_data['segment_number']
        lx = file_data['lx']
        ly = file_data['ly']
        rx = file_data['rx']
        ry = file_data['ry']
        label = file_data['label']
        Fs = file_data['Fs']
        
        st.markdown('#### Information')
        st.write("🧑 Subject: ", subject)
        st.write("📖 Reading task: ", task)
        st.write("🩺 Diagnosis: ", class_names[label])

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------
def update(num, x, y, speedup, line, x_max, y_max):
    line.set_data(x[:num*speedup]*x_max, y[:num*speedup]*y_max)
    line.axes.axis([0, x_max, y_max, 0])
    return line,

def plot_animation(img, x, y, Fs, interval, speedup, filename, dpi):
    x_max = np.shape(img)[1]
    y_max = np.shape(img)[0]
    
    plt.rcParams["animation.html"] = "jshtml"
    plt.rcParams['figure.dpi'] = 150  
    plt.ioff()
    fig, ax = plt.subplots()  # figsize=(10, 5)
    plt.imshow(img)
    line, = plt.plot(x*x_max, y*y_max, color='tab:blue')
    plt.box(False)
    plt.xticks([])
    plt.yticks([])
    ani = animation.FuncAnimation(fig, func=update, frames=len(x)//speedup, fargs=[x, y, speedup, line, x_max, y_max], 
                                  interval=interval, blit=True)
    ani.save(filename, dpi=dpi, fps=Fs//speedup)
    plt.close()
    
def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def create_altair_scanpath(image_path, aspect_ratio, fixations, forw_saccades, backw_saccades):
    
    # Plot stimulus image in the background -----------------------------------
    pil_image = Image.open(image_path)
    output = BytesIO()
    pil_image.save(output, format='PNG', dpi=(600, 600))
    base64_image = (
        "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
    )
    
    x_max, y_max = pil_image.size
    x_max = x_max * aspect_ratio
    y_max = y_max * aspect_ratio
    
    img_source = pd.DataFrame({"x": [0], "x2": [x_max], "y": [y_max], "y2": [0], "image": [base64_image]})
    bckgnd_img = (
        alt.Chart(img_source)
        .mark_image(width=x_max, height=y_max, align='left', baseline='bottom')
        .encode(
            alt.X("x:Q", scale=alt.Scale(domain=[0, x_max], nice=False)),
            alt.Y("y:Q", scale=alt.Scale(domain=[y_max, 0], nice=False)),
            x2="x2",
            y2="y2",
            url="image"
            )
        .properties(
            width=x_max,
            height=y_max,
            )
        )
    
    # Plot fixations data -----------------------------------------------------
    fixations_data = {"horizontal coordinate": fixations['x'] * x_max, 
                      "vertical coordinate": fixations['y'] * y_max, 
                      "horizontal component": fixations['x'], 
                      "vertical component": fixations['y'], 
                      "fixation duration in ms": fixations['dur'],
                      "order": [i+1 for i in range(len(fixations['x']))]}
    st.session_state.data = pd.DataFrame(fixations_data)
    fixations_data = st.session_state.data
    
    fixation_selector = alt.selection_point("point_selection")
            
    fixations_chart = (
        alt.Chart(fixations_data)
        .mark_circle(opacity=0.8)
        .encode(
            alt.X("horizontal coordinate:Q", scale=alt.Scale(domain=[0, x_max], nice=False), axis=alt.Axis(labels=False, title=None)),
            alt.Y("vertical coordinate:Q", scale=alt.Scale(domain=[y_max, 0], nice=False), axis=alt.Axis(labels=False, title=None)),
            # x="horizontal coordinate",
            # y="vertical coordinate",
            size=alt.Size("fixation duration in ms", scale=alt.Scale(range=[200, 600]), legend=None),
            color=alt.value("#1F77B4"),
            # color=alt.Color("fixation duration", legend=None),
            tooltip=["horizontal component", "vertical component", "fixation duration in ms"],
            fillOpacity=alt.condition(fixation_selector, alt.value(1), alt.value(0.7)),
        )
        .properties(
            width=x_max,
            height=y_max,
        )
        .add_params(fixation_selector)
    )
        
    fixations_sequence = alt.Chart(fixations_data).mark_text(baseline="middle").encode(
        alt.X("horizontal coordinate:Q", scale=alt.Scale(domain=[0, x_max], nice=False), axis=alt.Axis(labels=False, title=None)),
        alt.Y("vertical coordinate:Q", scale=alt.Scale(domain=[y_max, 0], nice=False), axis=alt.Axis(labels=False, title=None)),
        text="order:Q",
        color=alt.value("white"),
    )
    
    # Plot saccades data ------------------------------------------------------
    max_speed = np.max(np.concatenate((forw_saccades['speed'], np.abs(backw_saccades['speed']))))
    
    forw_saccades_df = pd.DataFrame({"x": forw_saccades['x'] * x_max,
                          "y": forw_saccades['y'] * y_max,
                          "x2": forw_saccades['x2'] * x_max,
                          "y2": forw_saccades['y2'] * y_max,
                          "speed": forw_saccades['speed'] / max_speed})
    if len(backw_saccades) > 0:
        backw_saccades_df = pd.DataFrame({"x": backw_saccades['x'] * x_max,
                              "y": backw_saccades['y'] * y_max,
                              "x2": backw_saccades['x2'] * x_max,
                              "y2": backw_saccades['y2'] * y_max,
                              "speed": np.abs(backw_saccades['speed']) / max_speed})
    else:
        backward_saccades_df = pd.DataFrame()
        
    saccades_df = pd.concat([forw_saccades_df, backw_saccades_df], axis=0, ignore_index=True)
    
    df_x = saccades_df.reset_index().melt(id_vars=["index"], 
                                               value_vars=['x','x2'], 
                                               var_name="x", 
                                               value_name="horizontal coordinate").sort_values(by=["index"]).rename(columns={"index": "line"})
    df_y = saccades_df.reset_index().melt(id_vars=["index"], 
                                               value_vars=['y','y2'], 
                                               var_name="y", 
                                               value_name="vertical coordinate").sort_values(by=["index"]).rename(columns={"index": "line"})
    df = pd.concat([df_x, df_y], axis=1)[["line", "horizontal coordinate", "vertical coordinate"]]
    df = df.loc[:,~df.columns.duplicated()].copy()
    saccades_data = pd.concat([df.reset_index().drop("index", axis=1), 
                                    np.repeat(saccades_df["speed"], 2).reset_index().drop("index", axis=1)], axis=1)
     
    saccades_chart = (
        alt.Chart(saccades_data)
        .mark_line()
        .encode(
            alt.X("horizontal coordinate:Q", scale=alt.Scale(domain=[0, x_max], nice=False), axis=alt.Axis(labels=False, title=None)),
            alt.Y("vertical coordinate:Q", scale=alt.Scale(domain=[y_max, 0], nice=False), axis=alt.Axis(labels=False, title=None)),
            color=alt.Color("speed", legend=None).scale(scheme="reds"),
            detail="line",
        )
        .properties(
            width=x_max,
            height=y_max,
        )
    )    
    return bckgnd_img, fixations_chart, fixations_sequence, saccades_chart
         
def create_scanpath(img, fixations, forw_saccades, backw_saccades):
    fig, ax = plt.subplots(dpi = 600)
    ax.imshow(img)

    x_max = np.shape(img)[1]
    y_max = np.shape(img)[0]

    max_speed = np.max(np.concatenate((forw_saccades['speed'], backw_saccades['speed'])))

    ax.scatter(x = fixations['x'] * x_max,
               y = fixations['y'] * y_max,
               s = (fixations['dur']) / 2,
               c = 'tab:blue',
               alpha = 0.8,
               edgecolors = 'none')

    # Connect the fixations with lines
    for i in range(len(forw_saccades['x'])):
        ax.arrow(forw_saccades['x'][i] * x_max,
                 forw_saccades['y'][i] * y_max,
                 (forw_saccades['dx'][i]) * x_max,
                 (forw_saccades['dy'][i]) * y_max,
                 length_includes_head = True,
                 head_width = 5,
                 head_length = 7,
                 color = adjust_lightness('tab:red', forw_saccades['speed'][i]/max_speed))
        
    for i in range(len(backw_saccades['x'])):
        ax.arrow(backw_saccades['x'][i] * x_max,
                 backw_saccades['y'][i] * y_max,
                 (backw_saccades['dx'][i]) * x_max,
                 (backw_saccades['dy'][i]) * y_max,
                 length_includes_head = True,
                 head_width = 5,
                 head_length = 7,
                 color = adjust_lightness('tab:red', backw_saccades['speed'][i]/max_speed))

    # Add numbers to each fixation in the scanpath
    for i, (x, y) in enumerate(zip(fixations['x'], fixations['y'])):
        ax.text(x * x_max,
                y * y_max,
                i+1,
                color = 'white',
                fontsize = 6,
                ha = 'center',
                va = 'center')

    ax.set_xlim([0, x_max])
    ax.set_ylim([y_max, 0])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Return the figure object
    return fig  

def n_min(a, N=3):
    s = sorted(enumerate(a), key=lambda x:x[1])[:N]
    # seperate values from indexes using zip
    z_idx, z_value = map(list, zip(*s))
    # yield each tuple of (values,indexes)
    return z_value, z_idx

def scanpath_analysis(x, y, t):
    
    fxs, fixations = fixation_detection(x, y, t, missing=-1, maxdist=0.05, mindur=48)
    
    fx_durs, fx_x, fx_y = [], [], []
    
    forw_x_start, forw_x_end, backw_x_start, backw_x_end = [], [], [], []
    forw_y_start, forw_y_end, backw_y_start, backw_y_end = [], [], [], []
    forw_speed, backw_speed = [], []
    
    for ix in range(1, len(fixations)):
        fstart0, fend0, dur0, fx0, fy0 = fixations[ix-1]
        fstart1, fend1, dur1, fx1, fy1 = fixations[ix]
    
        dx = fx1 - fx0
        dy = abs(fy1 - fy0)
        dt = fstart1 - fend0
    
        # forward saccades
        if dx > 0 and 2*dx > dy:
            forw_x_start.append(fx0)
            forw_x_end.append(fx1)
            forw_y_start.append(fy0)
            forw_y_end.append(fy1)
            forw_speed.append(dx/dt)
    
        # backward saccades
        if dx < 0 and 1*abs(dx) > dy:
            backw_x_start.append(fx0)
            backw_x_end.append(fx1)
            backw_y_start.append(fy0)
            backw_y_end.append(fy1)
            backw_speed.append(dx/dt)
            
        # fixation durations
        if ix == 1: fx_durs.append(dur0 / 1000)
        fx_durs.append(dur1 / 1000)
    
        if ix == 1: 
            fx_x.append(fx0)
            fx_y.append(fy0)
        fx_x.append(fx1)
        fx_y.append(fy1)
        
    fixations = {'x': np.array(fx_x), 'y': np.array(fx_y), 'dur': np.array(fx_durs)}
    forw_saccades = {'x': np.array(forw_x_start), 'y': np.array(forw_y_start), 
                     'x2': np.array(forw_x_end), 'y2': np.array(forw_y_end),
                     'dx': np.array([e_i - s_i for s_i, e_i in zip(forw_x_start, forw_x_end)]),
                     'dy': np.array([e_i - s_i for s_i, e_i in zip(forw_y_start, forw_y_end)]),
                     'speed': np.array(forw_speed),
                    }
    backw_saccades = {'x': np.array(backw_x_start), 'y': np.array(backw_y_start),
                      'x2': np.array(backw_x_end), 'y2': np.array(backw_y_end),
                     'dx': np.array([e_i - s_i for s_i, e_i in zip(backw_x_start, backw_x_end)]),
                     'dy': np.array([e_i - s_i for s_i, e_i in zip(backw_y_start, backw_y_end)]),
                     'speed': np.array(backw_speed),
                    }
    
    return fixations, forw_saccades, backw_saccades

def make_donut(input_response, input_text, input_color):
    if input_color == 'blue':
        chart_color = ['#29b5e8', '#155F7A']
    if input_color == 'green':
        chart_color = ['#27AE60', '#12783D']
    if input_color == 'orange':
        chart_color = ['#F39C12', '#875A12']
    if input_color == 'red':
        chart_color = ['#E74C3C', '#781F16']
      
    source = pd.DataFrame({
        "quantity": ['', input_text],
        "% value": [100-input_response, input_response]
    })
    source_bg = pd.DataFrame({
        "quantity": ['', input_text],
        "% value": [100, 0]
    })
      
    plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
        theta="% value",
        color= alt.Color("quantity:N",
                        scale=alt.Scale(
                            #domain=['A', 'B'],
                            domain=[input_text, ''],
                            # range=['#29b5e8', '#155F7A']),  # 31333F
                            range=chart_color),
                        legend=None),
    ).properties(width=130, height=130)
      
    text = plot.mark_text(align='center', color="#29b5e8", fontSize=24, fontWeight=700, fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
    plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
        theta="% value",
        color= alt.Color("quantity:N",
                        scale=alt.Scale(
                            # domain=['A', 'B'],
                            domain=[input_text, ''],
                            range=chart_color),  # 31333F
                        legend=None),
    ).properties(width=130, height=130)
    return plot_bg + plot + text

def create_container_with_color(id, parent_plh, color="#FFFFFF"):
    # todo: instead of color you can send in any css
    plh = parent_plh.container()
    html_code = """<div id = 'my_div_outer'></div>"""
    parent_plh.markdown(html_code, unsafe_allow_html=True)

   
    with plh:
        inner_html_code = """<div id = 'my_div_inner_%s'></div>""" % id
        plh.markdown(inner_html_code, unsafe_allow_html=True)

    ## applying style
    chat_plh_style = """
        <style>
            div[data-testid='stVerticalBlock']:has(div#my_div_inner_%s):not(:has(div#my_div_outer)) {
                background-color: %s;
                border-radius: 20px;
                padding: 10px;
            };
        </style>
        """
    chat_plh_style = chat_plh_style % (id, color)

    parent_plh.markdown(chat_plh_style, unsafe_allow_html=True)
    return plh

# -----------------------------------------------------------------------------
# COLUMNS
# -----------------------------------------------------------------------------

if uploaded_file is not None:
    # Process raw data --------------------------------------------------------   
    lx = (lx - norm_data[subject]['lx_min']) / (norm_data[subject]['lx_max'] - norm_data[subject]['lx_min'])
    ly = (ly - norm_data[subject]['ly_min']) / (norm_data[subject]['ly_max'] - norm_data[subject]['ly_min'])
    rx = (rx - norm_data[subject]['rx_min']) / (norm_data[subject]['rx_max'] - norm_data[subject]['rx_min'])
    ry = (ry - norm_data[subject]['ry_min']) / (norm_data[subject]['ry_max'] - norm_data[subject]['ry_min'])
    
    x = np.mean(np.stack((lx, rx), axis=0), axis=0)
    y = np.mean(np.stack((ly, ry), axis=0), axis=0)
    
    input_data = np.stack((x, y), axis=0)
    
    # Calculate features ------------------------------------------------------
    readTime = len(x) / Fs
    t = np.linspace(0, readTime*1000, len(x), 1/Fs*1000)  # Time in milliseconds
    
    success, active_read_time, fixation_intersection_coeff, saccade_variability, fixation_intersection_variability, fixation_fractal_dimension, fixation_count, fixation_total_dur, fixation_freq, fixation_avg_dur, saccade_count, saccade_total_dur, saccade_freq, saccade_avg_dur, total_read_time = feature_extraction(t, input_data[0,:], input_data[1,:], Fs)
    features = pd.DataFrame({'feature': ['active_read_time', 'fixation_intersection_coeff', 'saccade_variability', 'fixation_intersection_variability', 
                                         'fixation_fractal_dimension', 'fixation_count', 'fixation_total_dur',
                                         'fixation_freq', 'fixation_avg_dur', 'saccade_count', 'saccade_total_dur',
                                         'saccade_freq', 'saccade_avg_dur', 'total_read_time'],
                             'value': [active_read_time, fixation_intersection_coeff, saccade_variability, fixation_intersection_variability, 
                                       fixation_fractal_dimension, fixation_count, fixation_total_dur, 
                                       fixation_freq, fixation_avg_dur, saccade_count, saccade_total_dur, 
                                       saccade_freq, saccade_avg_dur, total_read_time],
                             'name': ['active reading time', 'fixation intersection coefficient', 
                                      'saccade variability', 'fixation intersection variability',
                                      'fixation fractal dimension', 'fixation count', 'fixation total duration',
                                      'fixation frequency', 'fixation average duration', 'saccade count', 'saccade total duration',
                                      'saccade frequency', 'saccade average duration', 'total reading time'],
        })
    
    fname = data_params.manifest_path + "feature-distribution/" + "train" + '_feature_distribution.pkl'
    # train_stats = scipy.io.loadmat(data_params.manifest_path + "dataset-distribution/" + "train" + '_feature_distribution.mat')
    with open(fname, 'rb') as f:
        train_stats = pickle.load(f)


st.markdown('### 🧠 Dyslexia Detection from Eye-Tracking Data')
st.divider()

# -----------------------------------------------------------------------------
# FIRST ROW
# -----------------------------------------------------------------------------
if uploaded_file is not None:
    x = input_data[0,:]
    y = input_data[1,:]
    
    readTime = len(x) / Fs
    time = np.linspace(0, readTime, len(x))
    
    hvc_help_text = """The graph represents horizontal and vertical components  
    of the eye movements."""
    st.markdown('##### 📈 Horizontal and vertical components', help=hvc_help_text)
    
    components_data = {"time in seconds": time, "horizontal": input_data[0,:], "vertical": input_data[1,:]}
    components_data = pd.DataFrame(components_data)
    components_data = components_data.reset_index().melt(id_vars=["time in seconds"], 
                                               value_vars=["horizontal", "vertical"],
                                               var_name="component", value_name="relative screen position")
    
    selection = alt.selection(type="point", nearest=True, on="mouseover",
                              clear='mouseout', fields=["time in seconds"], 
                              empty=False)
    
    base = alt.Chart(components_data).encode(x="time in seconds:Q")
    lines = base.encode(x="time in seconds:Q").mark_line().encode(
        x="time in seconds:Q",
        y="relative screen position:Q",
        color=alt.Color("component:N").scale(scheme="category10"),
        opacity=alt.value(1),
        )
    points = lines.mark_point().transform_filter(selection)
    rule = base.transform_pivot(
        "component", value="relative screen position", groupby=["time in seconds"]
    ).mark_rule().encode(
        opacity=alt.condition(selection, alt.value(0.3), alt.value(0)),
        tooltip=["horizontal:Q", "vertical:Q", "time in seconds:Q"],
    ).add_params(selection)
    text = lines.mark_text(align="left", dx=5, dy=-10).encode(
        text=alt.condition(selection, alt.Text("relative screen position:Q", format='.2f'), alt.value(" ")),
        opacity=alt.value(1),
        )
    st.altair_chart(alt.layer(lines, points, rule).interactive(), use_container_width=True)
        
        
if uploaded_file is not None:        
    st.divider()
    st.markdown('#### Data analysis')

st.markdown('<style>#vg-tooltip-element{z-index: 1000051}</style>', unsafe_allow_html=True)    

# -----------------------------------------------------------------------------
# SECOND ROW
# -----------------------------------------------------------------------------

col1 = 3.7
col2 = 4.3
second_row_cols = st.columns((col1, col2), gap='medium')

with second_row_cols[0]:
    
    if uploaded_file is not None:
        c21 = st.container(height=580, border=True)
        c21.markdown('##### 🛠️ Hand-crafted features')
        c21.markdown('#####')
        
        c21.write('**Reading time:** %.1f seconds' % readTime)

with second_row_cols[1]:
    
    if uploaded_file is not None:
        
        c22 = st.container(height=580, border=True)
        fs_help_text = """Compare the features of the current eye-tracking sequence with the  
        feature distribution of the sequences used for training the AI model."""
        c22.markdown('##### 📊 Feature statistics', 
                     help=fs_help_text)
        # ---------------------------------------------------------------------
        if c21.checkbox('**Active reading time:** %.2f seconds' % active_read_time, value=True):
            c22.markdown('##### ')
            c22.markdown('##### Active reading times')
            
            active_read_time_bin_width = 0.05
            
            he_active_read_times = [x for x in train_stats["active_read_times"][class_names[0]] if not np.isnan(x)]
            ci_active_read_times = [x for x in train_stats["active_read_times"][class_names[1]] if not np.isnan(x)]
            
            fig_active_read_times = ff.create_distplot(
                [he_active_read_times, ci_active_read_times], 
                class_names, show_rug=False, curve_type='normal', bin_size=0.25)
            fig_active_read_times.update_layout(xaxis_title="active reading time in seconds", yaxis_title="frequency")
            fig_active_read_times.update_traces(marker_line_width=1, marker_line_color="black")
            fig_active_read_times.add_vrect(x0=active_read_time-active_read_time_bin_width, y0=0,
                               x1=active_read_time+active_read_time_bin_width, y1=1,
                               fillcolor="red", opacity=0.35, line_width=0)
            fig_active_read_times.add_trace(
                go.Scatter(
                    x=[active_read_time-active_read_time_bin_width, 
                       active_read_time-active_read_time_bin_width, 
                       active_read_time+active_read_time_bin_width, 
                       active_read_time+active_read_time_bin_width, 
                       active_read_time-active_read_time_bin_width], 
                    y=[0, 1.5, 1.5, 0, 0], 
                    fill="toself",
                    mode='lines',
                    name='',
                    text='active reading time in seconds = %.2f' %active_read_time,
                    opacity=0,
                )
            )
            c22.plotly_chart(fig_active_read_times, use_container_width=True)
            
        # ---------------------------------------------------------------------
        if c21.checkbox('**Fixation intersection coefficient:** %.2f' % fixation_intersection_coeff):
            c22.markdown('##### ')
            c22.markdown('##### Fixation intersection coefficients')
            
            fixation_intersection_coeff_bin_width = 0.1
            
            he_fixation_intersection_coeffs = [x for x in train_stats["fixation_intersection_coeffs"][class_names[0]] if not np.isnan(x)]
            ci_fixation_intersection_coeffs = [x for x in train_stats["fixation_intersection_coeffs"][class_names[1]] if not np.isnan(x)]
            
            fig_fixation_intersection_coeffs = ff.create_distplot(
                [he_fixation_intersection_coeffs, ci_fixation_intersection_coeffs], 
                class_names, show_rug=False, curve_type='normal', bin_size=1)
            fig_fixation_intersection_coeffs.update_layout(xaxis_title="fixation intersection coefficient", yaxis_title="frequency")
            fig_fixation_intersection_coeffs.update_traces(marker_line_width=1, marker_line_color="black")
            fig_fixation_intersection_coeffs.add_vrect(x0=fixation_intersection_coeff-fixation_intersection_coeff_bin_width, y0=0,
                                x1=fixation_intersection_coeff+fixation_intersection_coeff_bin_width, y1=1,
                                fillcolor="red", opacity=0.35, line_width=0)
            fig_fixation_intersection_coeffs.add_trace(
                go.Scatter(
                    x=[fixation_intersection_coeff-fixation_intersection_coeff_bin_width, 
                       fixation_intersection_coeff-fixation_intersection_coeff_bin_width, 
                       fixation_intersection_coeff+fixation_intersection_coeff_bin_width, 
                       fixation_intersection_coeff+fixation_intersection_coeff_bin_width, 
                       fixation_intersection_coeff-fixation_intersection_coeff_bin_width], 
                    y=[0, 0.9, 0.9, 0, 0], 
                    fill="toself",
                    mode='lines',
                    name='',
                    text='fixation intersection coefficient = %.2f' %fixation_intersection_coeff,
                    opacity=0,
                )
            )
            c22.plotly_chart(fig_fixation_intersection_coeffs, use_container_width=True)
            
        # ---------------------------------------------------------------------
        if c21.checkbox('**Saccade variability:** %.1f ms' % saccade_variability):
            c22.markdown('##### ')
            c22.markdown('##### Saccade variabilities')
            
            saccade_wariability_bin_width = 10
            
            he_saccade_variabilities = [x for x in train_stats["saccade_variabilities"][class_names[0]] if not np.isnan(x)]
            ci_saccade_variabilities = [x for x in train_stats["saccade_variabilities"][class_names[1]] if not np.isnan(x)]
            
            fig_saccade_variabilities = ff.create_distplot(
                [he_saccade_variabilities, ci_saccade_variabilities], 
                class_names, show_rug=False, curve_type='normal', bin_size=50)
            fig_saccade_variabilities.update_layout(xaxis_title="saccade variability in ms", yaxis_title="frequency")
            fig_saccade_variabilities.update_traces(marker_line_width=1, marker_line_color="black")
            fig_saccade_variabilities.add_vrect(x0=saccade_variability-saccade_wariability_bin_width, y0=0,
                                x1=saccade_variability+saccade_wariability_bin_width, y1=1,
                                fillcolor="red", opacity=0.35, line_width=0)
            fig_saccade_variabilities.add_trace(
                go.Scatter(
                    x=[saccade_variability-saccade_wariability_bin_width, 
                        saccade_variability-saccade_wariability_bin_width, 
                        saccade_variability+saccade_wariability_bin_width, 
                        saccade_variability+saccade_wariability_bin_width, 
                        saccade_variability-saccade_wariability_bin_width], 
                    y=[0, 0.009, 0.009, 0, 0], 
                    fill="toself",
                    mode='lines',
                    name='',
                    text='saccade variability in ms = %.1f' %saccade_variability,
                    opacity=0,
                )
            )
            c22.plotly_chart(fig_saccade_variabilities, use_container_width=True)
            
        # ---------------------------------------------------------------------
        if c21.checkbox('**Fixation intersection variability:** %.2f' % fixation_intersection_variability):
            c22.markdown('##### ')
            c22.markdown('##### Fixation intersection variabilities')
            
            fixation_intersection_variability_bin_width = 0.25
            
            he_fixation_intersection_variabilities = [x for x in train_stats["fixation_intersection_variabilities"][class_names[0]] if not np.isnan(x)]
            ci_fixation_intersection_variabilities = [x for x in train_stats["fixation_intersection_variabilities"][class_names[1]] if not np.isnan(x)]
            
            fig_fixation_intersection_variabilities = ff.create_distplot(
                [he_fixation_intersection_variabilities, ci_fixation_intersection_variabilities], 
                class_names, show_rug=False, curve_type='normal', bin_size=1)
            fig_fixation_intersection_variabilities.update_layout(xaxis_title="number of forward saccades", yaxis_title="frequency")
            fig_fixation_intersection_variabilities.update_traces(marker_line_width=1, marker_line_color="black")
            fig_fixation_intersection_variabilities.add_vrect(x0=fixation_intersection_variability-fixation_intersection_variability_bin_width, y0=0,
                               x1=fixation_intersection_variability+fixation_intersection_variability_bin_width, y1=1,
                               fillcolor="red", opacity=0.35, line_width=0)
            fig_fixation_intersection_variabilities.add_trace(
                go.Scatter(
                    x=[fixation_intersection_variability-fixation_intersection_variability_bin_width, 
                       fixation_intersection_variability-fixation_intersection_variability_bin_width, 
                       fixation_intersection_variability+fixation_intersection_variability_bin_width, 
                       fixation_intersection_variability+fixation_intersection_variability_bin_width, 
                       fixation_intersection_variability-fixation_intersection_variability_bin_width], 
                    y=[0, 0.6, 0.6, 0, 0], 
                    fill="toself",
                    mode='lines',
                    name='',
                    text='fixation intersection variability = %.2f' %fixation_intersection_variability,
                    opacity=0,
                )
            )
            c22.plotly_chart(fig_fixation_intersection_variabilities, use_container_width=True)
            
        # ---------------------------------------------------------------------
        if c21.checkbox('**Fixation fractal dimension:**  %.2f' % fixation_fractal_dimension):
            c22.markdown('##### ')
            c22.markdown('##### Fixation fractal dimensions')
            
            fixation_fractal_dimension_bin_width = 0.01
            
            he_fixation_fractal_dimensions = [x for x in train_stats["fixation_fractal_dimensions"][class_names[0]] if not np.isnan(x)]
            ci_fixation_fractal_dimensions = [x for x in train_stats["fixation_fractal_dimensions"][class_names[1]] if not np.isnan(x)]
            
            fig_fixation_fractal_dimensions = ff.create_distplot(
                [he_fixation_fractal_dimensions, ci_fixation_fractal_dimensions], 
                class_names, show_rug=False, curve_type='normal', bin_size=0.05)
            fig_fixation_fractal_dimensions.update_layout(xaxis_title="fixation fractal dimension", yaxis_title="frequency")
            fig_fixation_fractal_dimensions.update_traces(marker_line_width=1, marker_line_color="black")
            fig_fixation_fractal_dimensions.add_vrect(x0=fixation_fractal_dimension-fixation_fractal_dimension_bin_width, y0=0,
                               x1=fixation_fractal_dimension+fixation_fractal_dimension_bin_width, y1=1,
                               fillcolor="red", opacity=0.35, line_width=0)
            fig_fixation_fractal_dimensions.add_trace(
                go.Scatter(
                    x=[fixation_fractal_dimension-fixation_fractal_dimension_bin_width, 
                       fixation_fractal_dimension-fixation_fractal_dimension_bin_width, 
                       fixation_fractal_dimension+fixation_fractal_dimension_bin_width, 
                       fixation_fractal_dimension+fixation_fractal_dimension_bin_width, 
                       fixation_fractal_dimension-fixation_fractal_dimension_bin_width], 
                    y=[0, 9, 9, 0, 0], 
                    fill="toself",
                    mode='lines',
                    name='',
                    text='fixation fractal dimension = %.2f' %fixation_fractal_dimension,
                    opacity=0,
                )
            )
            c22.plotly_chart(fig_fixation_fractal_dimensions, use_container_width=True)
            
        # ---------------------------------------------------------------------
        if c21.checkbox('**Fixation count:** %d' % fixation_count):
            c22.markdown('##### ')
            c22.markdown('##### Fixation counts')
            
            fixation_count_bin_width = 0.5
            
            he_fixation_counts = [x for x in train_stats["fixation_counts"][class_names[0]] if not np.isnan(x)]
            ci_fixation_counts = [x for x in train_stats["fixation_counts"][class_names[1]] if not np.isnan(x)]
            
            fig_fixation_counts = ff.create_distplot(
                [he_fixation_counts, ci_fixation_counts], 
                class_names, show_rug=False, curve_type='normal', bin_size=1)
            fig_fixation_counts.update_layout(xaxis_title="fixation count", yaxis_title="frequency")
            fig_fixation_counts.update_traces(marker_line_width=1, marker_line_color="black")
            fig_fixation_counts.add_vrect(x0=fixation_count-fixation_count_bin_width, y0=0,
                               x1=fixation_count+fixation_count_bin_width, y1=1,
                               fillcolor="red", opacity=0.35, line_width=0)
            fig_fixation_counts.add_trace(
                go.Scatter(
                    x=[fixation_count-fixation_count_bin_width, 
                       fixation_count-fixation_count_bin_width, 
                       fixation_count+fixation_count_bin_width, 
                       fixation_count+fixation_count_bin_width, 
                       fixation_count-fixation_count_bin_width], 
                    y=[0, 0.16, 0.16, 0, 0], 
                    fill="toself",
                    mode='lines',
                    name='',
                    text='fixation count = %d' %fixation_count,
                    opacity=0,
                )
            )
            c22.plotly_chart(fig_fixation_counts, use_container_width=True)
        
        # ---------------------------------------------------------------------
        if c21.checkbox('**Fixation total duration:** %.1f ms' % fixation_total_dur):
            c22.markdown('##### ')
            c22.markdown('##### Fixation total durations')
            
            fixation_total_dur_bin_width = 50
            
            he_fixation_total_durs = [x for x in train_stats["fixation_total_durs"][class_names[0]] if not np.isnan(x)]
            ci_fixation_total_durs = [x for x in train_stats["fixation_total_durs"][class_names[1]] if not np.isnan(x)]
            
            fig_fixation_total_durs = ff.create_distplot(
                [he_fixation_total_durs, ci_fixation_total_durs], 
                class_names, show_rug=False, curve_type='normal', bin_size=250)
            fig_fixation_total_durs.update_layout(xaxis_title="fixation total duration in ms", yaxis_title="frequency")
            fig_fixation_total_durs.update_traces(marker_line_width=1, marker_line_color="black")
            fig_fixation_total_durs.add_vrect(x0=fixation_total_dur-fixation_total_dur_bin_width, y0=0,
                               x1=fixation_total_dur+fixation_total_dur_bin_width, y1=1,
                               fillcolor="red", opacity=0.35, line_width=0)
            fig_fixation_total_durs.add_trace(
                go.Scatter(
                    x=[fixation_total_dur-fixation_total_dur_bin_width, 
                       fixation_total_dur-fixation_total_dur_bin_width, 
                       fixation_total_dur+fixation_total_dur_bin_width, 
                       fixation_total_dur+fixation_total_dur_bin_width, 
                       fixation_total_dur-fixation_total_dur_bin_width], 
                    y=[0, 0.001, 0.001, 0, 0], 
                    fill="toself",
                    mode='lines',
                    name='',
                    text='fixation total duration in ms = %.1f' %fixation_total_dur,
                    opacity=0,
                )
            )
            c22.plotly_chart(fig_fixation_total_durs, use_container_width=True)
            
        # ---------------------------------------------------------------------
        if c21.checkbox('**Fixation frequency:** %.2f Hz' % fixation_freq):
            c22.markdown('##### ')
            c22.markdown('##### Fixation frequencies')
            
            fixation_freq_bin_width = 0.05
                            
            he_fixation_freqs = [x for x in train_stats["fixation_freqs"][class_names[0]] if not np.isnan(x)]
            ci_fixation_freqs = [x for x in train_stats["fixation_freqs"][class_names[1]] if not np.isnan(x)]
            
            fig_fixation_freqs = ff.create_distplot(
                [he_fixation_freqs, ci_fixation_freqs], 
                class_names, show_rug=False, curve_type='normal', bin_size=0.25)
            fig_fixation_freqs.update_layout(xaxis_title="fixation frequency in Hz", yaxis_title="frequency")
            fig_fixation_freqs.update_traces(marker_line_width=1, marker_line_color="black")
            fig_fixation_freqs.add_vrect(x0=fixation_freq-fixation_freq_bin_width, y0=0,
                               x1=fixation_freq+fixation_freq_bin_width, y1=1,
                               fillcolor="red", opacity=0.35, line_width=0)
            fig_fixation_freqs.add_trace(
                go.Scatter(
                    x=[fixation_freq-fixation_freq_bin_width, 
                       fixation_freq-fixation_freq_bin_width, 
                       fixation_freq+fixation_freq_bin_width, 
                       fixation_freq+fixation_freq_bin_width, 
                       fixation_freq-fixation_freq_bin_width], 
                    y=[0, 0.65, 0.65, 0, 0], 
                    fill="toself",
                    mode='lines',
                    name='',
                    text='fixation frequency in Hz = %.2f' %fixation_freq,
                    opacity=0,
                )
            )
            c22.plotly_chart(fig_fixation_freqs, use_container_width=True)
            
        # ---------------------------------------------------------------------
        if c21.checkbox('**Fixation average duration:** %.2f ms' % fixation_avg_dur):
            c22.markdown('##### ')
            c22.markdown('##### Fixation average durations')
            
            fixation_avg_dur_bin_width = 10
                            
            he_fixation_avg_durs = [x for x in train_stats["fixation_avg_durs"][class_names[0]] if not np.isnan(x)]
            ci_fixation_avg_durs = [x for x in train_stats["fixation_avg_durs"][class_names[1]] if not np.isnan(x)]
            
            fig_fixation_avg_durs = ff.create_distplot(
                [he_fixation_avg_durs, ci_fixation_avg_durs], 
                class_names, show_rug=False, curve_type='normal', bin_size=50)
            fig_fixation_avg_durs.update_layout(xaxis_title="fixation average duration in ms", yaxis_title="frequency")
            fig_fixation_avg_durs.update_traces(marker_line_width=1, marker_line_color="black")
            fig_fixation_avg_durs.add_vrect(x0=fixation_avg_dur-fixation_avg_dur_bin_width, y0=0,
                               x1=fixation_avg_dur+fixation_avg_dur_bin_width, y1=1,
                               fillcolor="red", opacity=0.35, line_width=0)
            fig_fixation_avg_durs.add_trace(
                go.Scatter(
                    x=[fixation_avg_dur-fixation_avg_dur_bin_width, 
                       fixation_avg_dur-fixation_avg_dur_bin_width, 
                       fixation_avg_dur+fixation_avg_dur_bin_width, 
                       fixation_avg_dur+fixation_avg_dur_bin_width, 
                       fixation_avg_dur-fixation_avg_dur_bin_width], 
                    y=[0, 0.01, 0.01, 0, 0], 
                    fill="toself",
                    mode='lines',
                    name='',
                    text='fixation average duration in ms = %.2f' %fixation_avg_dur,
                    opacity=0,
                )
            )
            c22.plotly_chart(fig_fixation_avg_durs, use_container_width=True)        
                
            
        # ---------------------------------------------------------------------
        if c21.checkbox('**Saccade count:** %d' % saccade_count):
            c22.markdown('##### ')
            c22.markdown('##### Saccade counts')
            
            saccade_count_bin_width = 0.25
            
            he_saccade_counts = [x for x in train_stats["saccade_counts"][class_names[0]] if not np.isnan(x)]
            ci_saccade_counts = [x for x in train_stats["saccade_counts"][class_names[1]] if not np.isnan(x)]
            
            fig_saccade_counts = ff.create_distplot(
                [he_saccade_counts, ci_saccade_counts], 
                class_names, show_rug=False, curve_type='normal', bin_size=1)
            fig_saccade_counts.update_layout(xaxis_title="saccade count", yaxis_title="frequency")
            fig_saccade_counts.update_traces(marker_line_width=1, marker_line_color="black")
            fig_saccade_counts.add_vrect(x0=saccade_count-saccade_count_bin_width, y0=0,
                                x1=saccade_count+saccade_count_bin_width, y1=1,
                                fillcolor="red", opacity=0.35, line_width=0)
            fig_saccade_counts.add_trace(
                go.Scatter(
                    x=[saccade_count-saccade_count_bin_width, 
                        saccade_count-saccade_count_bin_width, 
                        saccade_count+saccade_count_bin_width, 
                        saccade_count+saccade_count_bin_width, 
                        saccade_count-saccade_count_bin_width], 
                    y=[0, 0.16, 0.16, 0, 0],
                    fill="toself",
                    mode='lines',
                    name='',
                    text='number of saccades = %d' %saccade_count,
                    opacity=0,
                )
            )
            c22.plotly_chart(fig_saccade_counts, use_container_width=True)
                
        # ---------------------------------------------------------------------
        if c21.checkbox('**Saccade total duration:** %.1f ms' % saccade_total_dur):
            c22.markdown('##### ')
            c22.markdown('##### Median backward saccade speeds')
            
            saccade_total_dur_bin_width = 25
            
            he_saccade_total_durs = [x for x in train_stats["saccade_total_durs"][class_names[0]] if not np.isnan(x)]
            ci_saccade_total_durs = [x for x in train_stats["saccade_total_durs"][class_names[1]] if not np.isnan(x)]
            
            fig_saccade_total_durs = ff.create_distplot(
                [he_saccade_total_durs, ci_saccade_total_durs], 
                class_names, show_rug=False, curve_type='normal', bin_size=100)
            fig_saccade_total_durs.update_layout(xaxis_title="saccade total duration in ms", yaxis_title="frequency")
            fig_saccade_total_durs.update_traces(marker_line_width=1, marker_line_color="black")
            fig_saccade_total_durs.add_vrect(x0=saccade_total_dur-saccade_total_dur_bin_width, y0=0,
                               x1=saccade_total_dur+saccade_total_dur_bin_width, y1=1,
                               fillcolor="red", opacity=0.35, line_width=0)
            fig_saccade_total_durs.add_trace(
                go.Scatter(
                    x=[saccade_total_dur-saccade_total_dur_bin_width, 
                       saccade_total_dur-saccade_total_dur_bin_width, 
                       saccade_total_dur+saccade_total_dur_bin_width, 
                       saccade_total_dur+saccade_total_dur_bin_width, 
                       saccade_total_dur-saccade_total_dur_bin_width], 
                    y=[0, 0.001, 0.001, 0, 0], 
                    fill="toself",
                    mode='lines',
                    name='',
                    text='saccade total duration in ms = %.1f' %saccade_total_dur,
                    opacity=0,
                )
            )
            c22.plotly_chart(fig_saccade_total_durs, use_container_width=True)
                
        # ---------------------------------------------------------------------
        if c21.checkbox('**Saccade frequency:** %.2f Hz' % saccade_freq):
            c22.markdown('##### ')
            c22.markdown('##### Saccade frequencies')
            
            saccade_freq_bin_width = 0.05
            
            he_saccade_freqs = [x for x in train_stats["saccade_freqs"][class_names[0]] if not np.isnan(x)]
            ci_saccade_freqs = [x for x in train_stats["saccade_freqs"][class_names[1]] if not np.isnan(x)]
            
            fig_saccade_freqs = ff.create_distplot(
                [he_saccade_freqs, ci_saccade_freqs], 
                class_names, show_rug=False, curve_type='normal', bin_size=0.25)
            fig_saccade_freqs.update_layout(xaxis_title="saccade frequency in Hz", yaxis_title="frequency")
            fig_saccade_freqs.update_traces(marker_line_width=1, marker_line_color="black")
            fig_saccade_freqs.add_vrect(x0=saccade_freq-saccade_freq_bin_width, y0=0,
                               x1=saccade_freq+saccade_freq_bin_width, y1=1,
                               fillcolor="red", opacity=0.35, line_width=0)
            fig_saccade_freqs.add_trace(
                go.Scatter(
                    x=[saccade_freq-saccade_freq_bin_width, 
                       saccade_freq-saccade_freq_bin_width, 
                       saccade_freq+saccade_freq_bin_width, 
                       saccade_freq+saccade_freq_bin_width, 
                       saccade_freq-saccade_freq_bin_width], 
                    y=[0, 0.7, 0.7, 0, 0], 
                    fill="toself",
                    mode='lines',
                    name='',
                    text='saccade frequency in Hz = %.2f' %saccade_freq,
                    opacity=0,
                )
            )
            c22.plotly_chart(fig_saccade_freqs, use_container_width=True)
                
        # ---------------------------------------------------------------------
        if c21.checkbox('**Saccade average duration:** %.2f ms' % saccade_avg_dur):
            c22.markdown('##### ')
            c22.markdown('##### Median backward saccade speeds')
            
            saccade_avg_dur_bin_width = 1
            
            he_saccade_avg_durs = [x for x in train_stats["saccade_avg_durs"][class_names[0]] if not np.isnan(x)]
            ci_saccade_avg_durs = [x for x in train_stats["saccade_avg_durs"][class_names[1]] if not np.isnan(x)]
            
            fig_saccade_avg_durs = ff.create_distplot(
                [he_saccade_avg_durs, ci_saccade_avg_durs], 
                class_names, show_rug=False, curve_type='normal', bin_size=5)
            fig_saccade_avg_durs.update_layout(xaxis_title="saccade average duration in ms", yaxis_title="frequency")
            fig_saccade_avg_durs.update_traces(marker_line_width=1, marker_line_color="black")
            fig_saccade_avg_durs.add_vrect(x0=saccade_avg_dur-saccade_avg_dur_bin_width, y0=0,
                               x1=saccade_avg_dur+saccade_avg_dur_bin_width, y1=1,
                               fillcolor="red", opacity=0.35, line_width=0)
            fig_saccade_avg_durs.add_trace(
                go.Scatter(
                    x=[saccade_avg_dur-saccade_avg_dur_bin_width, 
                       saccade_avg_dur-saccade_avg_dur_bin_width, 
                       saccade_avg_dur+saccade_avg_dur_bin_width, 
                       saccade_avg_dur+saccade_avg_dur_bin_width, 
                       saccade_avg_dur-saccade_avg_dur_bin_width], 
                    y=[0, 0.1, 0.1, 0, 0], 
                    fill="toself",
                    mode='lines',
                    name='',
                    text='saccade average duration in ms = %.2f' %saccade_avg_dur,
                    opacity=0,
                )
            )
            c22.plotly_chart(fig_saccade_avg_durs, use_container_width=True)