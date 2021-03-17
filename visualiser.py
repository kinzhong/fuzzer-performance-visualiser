#!/usr/bin/env python3

import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import toml
import scipy.stats
import plotly.graph_objs as go
import webcolors
import datetime
import os
import errno
import sys


def main():

    if len(sys.argv) < 2:
        print("Usage: python3 visualiser.py [TOML_FILE]")
        sys.exit()

    file_loc = sys.argv[1]

    if not os.path.isfile(file_loc):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_loc)

    toml_dict = toml.load(file_loc)
    misc = toml_dict['misc']

    #Assigning configurations
    global bucket
    global confidence
    global out_dir
    global ylabel
    global file_postfix
    global project
    global max_time
    global x_axis_increment
    global y_axis_log_scale
    global no_legend
    global xlabel       

    bucket = misc['bucket']
    confidence = misc['confidence_lvl']
    out_dir = misc['out_dir']
    ylabel = misc['ylabel']
    file_postfix = misc["file_postfix"]
    project = misc["project"]
    max_time = misc['max_time']
    x_axis_increment = misc['x_axis_increment']
    y_axis_log_scale = misc['y_axis_log_scale']
    no_legend = misc["no_legend"]

    #NOT INCLUDED IN TOML FILE
    xlabel = "Time"

    print("Generating visualisations from " + file_loc)

    avg_df_list = []
    fuzzers = toml_dict['fuzzers']

    for fuzzer_name in fuzzers:        
        df = getAvg(fuzzer_name, fuzzers)
        df = discardPastTimeEntries(df)
        df = changeTimeBucket(df, bucket)
        df = changeTimeFormat(df)        
        avg_df_list.append( (fuzzer_name, fuzzers[fuzzer_name], df) )
    
    #Generate chart with Confidence
    fig = getChartWithConfidence(avg_df_list)

    #Generate chart with No Confidence
    fig2 = getChartWithNoConfidence(avg_df_list)

    #Save Results
    saveCharts(fig, fig2)
    print("Visualisations are saved in " + out_dir)

#Data Processing Functions

def discardPastTimeEntries(df):
    #max_time is in hours
    
    max_seconds = max_time * 60 * 60
    i=0
    for index, row in df.iterrows():
        if (row["time"] > max_seconds):
            df = df.loc[:index-1]
            i = index
            break
    
    
    new_data = pd.DataFrame(df[-1:].values, columns=df.columns)
    df = df.append(new_data)
    df = df.reset_index(drop=True)
    df.loc[df.index[-1], "time"] = max_seconds
    
    return df

def changeTimeFormat(df):
    
    for index in df.index:
        
        minutes, seconds = divmod(int(df.loc[index,"time"]), 60)
        hours, minutes = divmod(minutes, 60)
        
        days = 1
        additional_days, hours = divmod(hours, 24)
        days += additional_days
        
        months = 1
        additional_months, days = divmod(days, 30)
        months += additional_months
        df.loc[index, "time"] = datetime.datetime(2000,months,days,hours, minutes, seconds)
    
    return df

def meanConfidenceInterval(data, confidence_level):
    ar = 1.0 * np.array(data)
    n = len(ar)
    standard_error = scipy.stats.sem(ar)
    mean = np.mean(ar)
    h = standard_error * scipy.stats.t.ppf((1+confidence_level)/2., n-1)
    return mean, mean-h, mean+h

def changeTimeBucket(df, bucket):
    
    #Current df uses sec 
    if bucket == "sec":
        return df
    
    last_entry = df.loc[len(df)-1]
    highestTime = last_entry["time"]
    
    if (bucket == "min"):
        bucket_in_sec = 60
    elif (bucket == "hour"):
        bucket_in_sec = 60*60

    nextTime = 0
    data = []
    
    for index, row in df.iterrows():
        
        currentTime = row["time"]
        
        #print("Current Time: " + currentTime)
        if (currentTime == nextTime):
            data.append(row)
            nextTime += bucket_in_sec
            #print(type(row)
        elif (currentTime > nextTime):
            temp_row = df.loc[index-1].copy(deep=True)
            new_time = currentTime - (currentTime%bucket_in_sec)
            temp_row["time"] = new_time
            data.append(temp_row)
            nextTime = new_time + bucket_in_sec
    
    new_df = pd.DataFrame(data, columns=df.columns)
    new_df = new_df.reset_index(drop=True)

    return new_df

def getAvg(fuzzer, fuzzers):
    
    curr_time = 0;
    df_list = []
    sample_count = 0
    
    #Load data
    for datafile in fuzzers[fuzzer]['data_files']:
        path = datafile
        path = path.replace("/","\\")
        temp = pd.read_csv(path, header = None, names = ["time", "edge"], sep=':')
        sample_count+=1
        df_list.append([0,temp])
    
    #Get highest time
    highest_time = 0;
    for item in df_list:
        df = item[1]
        last_entry = df.loc[len(df)-1]
        if (highest_time < last_entry["time"]):
            highest_time = last_entry["time"]
        
        
        #avg_df = avg_df.append(item.loc[0])
    
    avg_i = 0
    avg_df = pd.DataFrame({'time': [], 'mean_edge': [], 'upper_bound': [], 'lower_bound': []})
    
    while(curr_time <= highest_time):

        #get all edges/entries for current time and remove all entries with the same time as curr_time
        data = []
        for item in df_list:
            curr_no = item[0]
            df = item[1]
            
            if (df.size > 0):
                top_entry = df.loc[0]

                if (top_entry["time"] == curr_time):
                    data.append(top_entry["edge"])
                    item[0] = top_entry["edge"]
                    item[1] = df.iloc[1:]
                    item[1] = item[1].reset_index(drop=True)

                else:
                    data.append(curr_no)
            else:
                data.append(curr_no)
         
        #Append to avg_df
        t = meanConfidenceInterval(data, confidence)
        avg_df.loc[avg_i] = [curr_time, t[0], t[1], t[2]]
        avg_i += 1
        
        if (curr_time == highest_time):
            break
        
        #Get next smallest time using updated df_list
        smallest = highest_time
        for item in df_list:
            df = item[1]
            if (df.size > 0):
                top_entry = df.loc[0]
                if (top_entry["time"] < smallest):
                    smallest = top_entry["time"]
        
        curr_time = smallest
    
    return avg_df



#Plotly Chart Functions

def getXAxes(first, last, increase_rate):
    
    value_list = []
    text_list = []
    curr = first

    if (increase_rate < datetime.timedelta(days = 1)):
        
        #if the increase rate is less than a day, the graph will plot in hourly
        no_of_days = 0
        i = first.hour
        
        while(curr <= last):

            if (i >= 24):
                no_of_days +=1
                i = i - 24
            
            value_list.append(curr)
            text_list.append(str(curr.hour+24*no_of_days) + "h")
            curr = curr + increase_rate
            
            i += increase_rate.seconds/60/60
    else:
        raise Exception("Invalid increase_rate")
        
    return value_list, text_list

def getChartWithConfidence(avg_df_list):
    
    fig = go.Figure()
    
    
    for df_tuple in avg_df_list:
        
        fuzzer_name = df_tuple[0]
        fuzzer_details = df_tuple[1]
        df = df_tuple[2]
        
        # dash options include “solid”, “dot”, “dash”, “longdash”, “dashdot”, or “longdashdot”
        dash_option = fuzzer_details['line_style']
        line_color = fuzzer_details['line_color']
        confidence_band_color = fuzzer_details['confidence_band_color']
        
        fig.add_trace(  
            go.Scatter(
                name=fuzzer_name,
                x=df["time"],
                y=df["mean_edge"],
                mode='lines',
                line=dict(color=line_color, dash=dash_option)
            )
        )
        fig.add_trace(  
            go.Scatter(
                name='Upper Bound',
                x=df["time"],
                y=df["upper_bound"],
                hoverinfo='none',  #Turns off hover for bounds
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            )
        )
        fig.add_trace(  
            go.Scatter(
                name='Lower Bound',
                x=df["time"],
                y=df["lower_bound"],
                hoverinfo='none',   #Turns off hover for bounds
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor=confidence_band_color,
                fill='tonexty',
                showlegend=False
            )
        )

    
    fig.update_layout(
        legend = dict(bgcolor = 'white'),    #legend fill color
        paper_bgcolor = "white",             #background color
        plot_bgcolor = "white",              #plot area color
        yaxis_title=ylabel,
        xaxis_title=xlabel,
        #title=chart_title,
        hovermode="x",
        showlegend = not no_legend
    )
    
    if (y_axis_log_scale):
        fig.update_yaxes(type="log")
    
    #first is the start point (0,0) and thus will not change
    first = datetime.datetime(2000,1,1,0, 0, 0)
    
    #Get highest time
    highest_time = first;
    for item in avg_df_list:
        df = item[2]
        last_entry = df.loc[len(df)-1]
        if (highest_time < last_entry["time"]):
            highest_time = last_entry["time"]
    
    
    last = highest_time
    increase = datetime.timedelta(hours = x_axis_increment)
    
    
    #if the increase rate is less than a day, the graph will plot in hourly
    t = getXAxes(first, last, increase)
    
    
    fig.update_xaxes(
        tickmode = 'array',
        tickvals = t[0],
        ticktext = t[1],
        showline = True,
        linecolor='black',
        linewidth = 2,
        
        ticks="outside",
        tickwidth = 3,
        tickcolor='black',
        ticklen=5
    )
    
    fig.update_yaxes(
        showline = True,
        linecolor='black',
        linewidth = 2,
        
        ticks="outside",
        tickwidth = 3,
        tickcolor='black',
        ticklen=5
    )
    
    return fig

def getChartWithNoConfidence(avg_df_list):
    
    fig = go.Figure()
    
    
    for df_tuple in avg_df_list:
        
        fuzzer_name = df_tuple[0]
        fuzzer_details = df_tuple[1]
        df = df_tuple[2]
        
        # dash options include “solid”, “dot”, “dash”, “longdash”, “dashdot”, or “longdashdot”
        dash_option = fuzzer_details['line_style']
        line_color = fuzzer_details['line_color']
        confidence_band_color = fuzzer_details['confidence_band_color']
        
        fig.add_trace(  
            go.Scatter(
                name=fuzzer_name,
                x=df["time"],
                y=df["mean_edge"],
                mode='lines',
                line=dict(color=line_color, dash=dash_option)
            )
        )
        

    
    fig.update_layout(
        legend = dict(bgcolor = 'white'),    #legend fill color
        paper_bgcolor = "white",             #background color
        plot_bgcolor = "white",              #plot area color
        yaxis_title=ylabel,
        xaxis_title=xlabel,
        #title=chart_title,
        hovermode="x",
        showlegend = not no_legend
    )
    
    if (y_axis_log_scale):
        fig.update_yaxes(type="log")

    
    #first is the start point (0,0) and thus will not change
    first = datetime.datetime(2000,1,1,0, 0, 0)
    
    #Get highest time
    highest_time = first;
    for item in avg_df_list:
        df = item[2]
        last_entry = df.loc[len(df)-1]
        if (highest_time < last_entry["time"]):
            highest_time = last_entry["time"]
    
    
    last = highest_time
    increase = datetime.timedelta(hours = x_axis_increment)
    
    
    #if the increase rate is less than a day, the graph will plot in hourly
    t = getXAxes(first, last, increase)
    
    
    fig.update_xaxes(
        tickmode = 'array',
        tickvals = t[0],
        ticktext = t[1],
        
        ticks="outside",
        tickwidth = 3,
        tickcolor='black',
        ticklen=5,
        
        showline = True,
        linecolor='black',
        linewidth = 2
    )
        
    
    fig.update_yaxes(
        showline = True,
        linecolor='black',
        linewidth = 2,
        
        ticks="outside",
        tickwidth = 3,
        tickcolor='black',
        ticklen=5,
    )
    
    return fig

def saveCharts(fig, fig2):

    #Saving results
    current_dir = os.getcwd()
    path = os.path.join(current_dir, out_dir) 

    if not os.path.exists(path):
        os.makedirs(path)

    os.chdir(path)

    if os.path.exists(project+file_postfix+'.html'):
      os.remove(project+file_postfix+'.html')
    fig.write_html(project+file_postfix+'.html')

    if os.path.exists(project+file_postfix+'.png'):
      os.remove(project+file_postfix+'.png')
    fig.write_image(project+file_postfix+'.png', scale=5)

    if os.path.exists(project+file_postfix+'.pdf'):
      os.remove(project+file_postfix+'.pdf')
    fig.write_image(project+file_postfix+'.pdf', scale=2)

    if os.path.exists(project+file_postfix+'-no-confidence.html'):
      os.remove(project+file_postfix+'-no-confidence.html')
    fig2.write_html(project+file_postfix+'-no-confidence.html')

    if os.path.exists(project+file_postfix+'-no-confidence.png'):
      os.remove(project+file_postfix+'-no-confidence.png')
    fig2.write_image(project+file_postfix+'-no-confidence.png', scale=5)

    if os.path.exists(project+file_postfix+'-no-confidence.pdf'):
      os.remove(project+file_postfix+'-no-confidence.pdf')
    fig2.write_image(project+file_postfix+'-no-confidence.pdf', scale=2)

    os.chdir(current_dir)

if __name__ == '__main__':
    main()
