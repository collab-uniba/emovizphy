"""
 Please note that this script uses scripts released by Taylor et al.
 that you can find here: https://github.com/MITMediaLabAffectiveComputing/eda-explorer

  Taylor, Sara et al. “Automatic identification of artifacts in electrodermal activity data.”
  Annual International Conference of the IEEE Engineering in Medicine and Biology Society.
  IEEE Engineering in Medicine and Biology Society.
  Annual International Conference vol. 2015 (2015): 1934-7.
  doi:10.1109/EMBC.2015.7318762
"""

import configparser
import datetime
import os
import re
import shutil
import zipfile
from datetime import date as dt
import pandas as pd
import panel as pn
from bokeh.models import Span
from scipy.stats import rankdata

from visualization_utils import (create_directories_session_data,
                                 create_directories_session_popup,
                                 create_fig_line, get_popup, read_param_EDA,
                                 save_data_filtered, create_directories_session_data_csv)

from avroPreprocess import Avro_to_Json_Convert as avro
from avroPreprocess import Json_to_Csv_Convert as json
from bokeh.models import Rect 
from bokeh.models import ColumnDataSource


# ================================ #
# Definition of panels and widgets #
# ================================ #

# Bokeh panels for eda, hr, and acc
bokeh_pane_eda = pn.pane.Bokeh(visible=False, sizing_mode="stretch_both")
bokeh_pane_hr = pn.pane.Bokeh(visible=False, sizing_mode="stretch_both")
bokeh_pane_acc = pn.pane.Bokeh(visible=False, sizing_mode="stretch_both")

# Panel widgets
text_title_student = pn.widgets.StaticText()
text_title_day = pn.widgets.StaticText()
text_title_session = pn.widgets.StaticText()


# Selecting the directory
selected_path_directory = None
def handle_upload(event):
    if dir_input_btn.filename.endswith('.avro'):
        dir_input_btn.save('data.avro')
    else:
        dir_input_btn.save('data.zip')
    select_directory()

dir_input_btn = pn.widgets.FileInput(accept='.zip', sizing_mode="stretch_width")
dir_input_btn.param.watch(handle_upload, 'filename', onlychanged=True)


# ========================================================== #
# Initialization of global variables and the Panel framework #
# ========================================================== #

file_name_student = None
current_session = None # Timestamp of the selected session.
path_student = None # Path of the student.
path_days = None # Path of the student's workdays.
path_sessions = None # Path of the workday sessions.
sessions = [] # List of timestamps of the sessions.
uploaded_file = None  # File's name uploaded

# Read config file
config_data = configparser.ConfigParser()
config_data.read("config.ini")
plot = config_data["PLOT"]

# Initialize Panel
pn.extension()


# ================= #
# Utility functions #
# ================= #

def select_directory():
    # this method allows the user to select the directory
    global selected_path_directory
    global text_title_student

    zipname = "./data.zip"
    dirname = "./data"
    with zipfile.ZipFile(zipname, 'r') as zip_ref:
        zip_ref.extractall(dirname)
    if dirname:
        selected_path_directory = dirname

    prepare_files(selected_path_directory)

    global file_name_student
    text_title_student.value = "Directory " + file_name_student + " selected"
    dir_input_btn = pn.widgets.FileInput(styles={'background': '#00A170'})

    dir_input_btn.aspect_ratio

    reset_widgets()

def reset_widgets():
    global button_visualize
    global bokeh_pane_eda, bokeh_pane_acc, bokeh_pane_hr
    global select
    button_visualize.disabled = True
    bokeh_pane_eda.visible = False
    bokeh_pane_acc.visible = False
    bokeh_pane_hr.visible = False

    global text_title_day, text_title_session
    text_title_day.value = ""
    text_title_session.value = ""

    select.disabled = True


def prepare_files(path):
    # this method copies and prepares the files in the temp
    global file_name_student
    # Get file directory
    file_name_student = os.path.basename(path)

    path_student = "./temp/" + file_name_student

    # split filename without extension
    uploaded_file = dir_input_btn.filename.split('.')[0]
    global path_days
    path_days = path_student + "/Sessions"

    # if exist temp folder, delete it
    if os.path.exists("./temp/"):
        # Delete Folders
        shutil.rmtree("./temp/")

    os.mkdir("./temp/")
    os.mkdir(path_student)

    if uploaded_file == "Data":
        shutil.copytree(path + "/Data", path_student + "/Data")
        shutil.copytree(path + "/Popup", path_student + "/Popup")
        create_directories_session_data(path_student)
        create_directories_session_popup(path_student)
    
    if uploaded_file != "Data" and os.path.exists(os.path.join("./data", uploaded_file)):
        shutil.copytree(os.path.join(path, uploaded_file), os.path.join(path_student, uploaded_file))
        create_directories_session_data_csv(path_student, uploaded_file)

    button_analyse.disabled = False

def visualize_session(date, session):
    global bokeh_pane_acc
    global bokeh_pane_eda
    global bokeh_pane_hr
    global progress_bar

    global plot

    # Check for missing signals in config
    signals = ["EDA", "HR", "ACC"]
    for s in signals:
        if s not in plot.keys():
            plot[s] = "0"

    path_session = "./temp/" + file_name_student + "/Sessions/" + date + "/" + session

    # x_range is used to move the graphs together on the x-axis
    x_range = None
    popup = None
    if os.path.exists(path_session + "/Popup"):
        popup = get_popup(path_session, date)

    # EDA
    if int(plot["EDA"]) == 1:
        bokeh_pane_eda.visible = True

        data = pd.read_csv(path_session + "/Data/data_eda_filtered.csv")

        data["time"] = pd.to_datetime(data["timestamp"])
        data["time"] = data["time"].values.astype("datetime64[s]")
        data["time"] = data["time"].dt.tz_localize("UTC").dt.tz_convert("Europe/Berlin")
        data["time"] = data["time"].dt.tz_localize(None)

        data = data[["time", "filtered_eda", "peaks"]]

        fig_eda = create_fig_line(
            data, "time", "filtered_eda", "Electrodermal Activity", "μS", "EDA", popup
        )

        # Add the peak markers to the figure
        peak_height = data["filtered_eda"].max() * 1.15
        data["peaks_plot"] = data["peaks"] * peak_height
        time_peaks = data[data["peaks_plot"] != 0]["time"]

        # if popup is not None:
        #     temp = popup.copy()
        #     temp["time"] = temp["time"].astype(str)
        #     temp["time"] = pd.to_datetime(temp["time"], format="%H:%M:%S").dt.time
        for t in time_peaks:
            # Arousal assignment
            arousal = None
            # considering pop-ups made before the peak
            if popup is not None:

                prev_popup = popup[popup["time"].dt.time < t.time()]
                prev_popup["time"] = prev_popup["time"].dt.time

                # considering only popups made in the previous 30 minutes
                if not prev_popup.empty:
                    # considering the last popup done in the previous 30 minutes
                    prev_popup = prev_popup.sort_values(by=["time"], ascending=False)
                    prev_popup.reset_index(inplace=True, drop=True)
                    flag = datetime.datetime.combine(
                        dt.today(), t.time()
                    ) - datetime.datetime.combine(
                        dt.today(), prev_popup.loc[0, "time"]
                    ) < datetime.timedelta(
                        minutes=30
                    )
                    if flag:
                        arousal = prev_popup.loc[0, "arousal"]

            if arousal is None:
                color = "#808080"  # Grey
            elif arousal == "Low 🧘‍♀" or arousal == "Low 😔":
                color = "#4DBD33"  # Green
            elif arousal == "Medium 😐":
                color = "#FF8C00"  # Orange
            else:
                color = "#FF0000"  # Red
            fig_eda.add_layout(
                Span(
                    location=t,
                    dimension="height",
                    line_color=color,
                    line_alpha=0.5,
                    line_width=1,
                )
            )

        if x_range is None:
            x_range = fig_eda.x_range

        fig_eda.x_range = x_range
        bokeh_pane_eda.object = fig_eda
    
        path_tags = os.path.join(path_session, "Data", "tags.csv")

        if os.path.isfile(path_tags) and os.stat(path_tags).st_size > 0:
            df = pd.read_csv(path_tags, header=None)
            timestamp_start = df.iloc[0, 0]
            timestamp_start = pd.to_datetime(int(timestamp_start), unit='s', utc=True).tz_convert("Europe/Berlin").tz_localize(None)
            eda_subset = data[(data["time"] == timestamp_start)]
            fig_eda.circle(x="time", y="filtered_eda", source=eda_subset, size=3, color="red")
            
            if len(df) >= 2:  # if there are start timestamp and end timestamp
                timestamp_end = df.iloc[1, 0]
                timestamp_end = pd.to_datetime(int(timestamp_end), unit='s', utc=True).tz_convert("Europe/Berlin").tz_localize(None)
                eda_subset_end = data[(data["time"] == timestamp_end)]
                fig_eda.circle(x="time", y="filtered_eda", source=eda_subset_end, size=3, color="red")

    # ACC
    if int(plot["ACC"]) == 1:

        bokeh_pane_acc.visible = True
        df_acc = pd.read_csv(path_session + "/Data/df_data_acc_filtered.csv")

        df_acc["time"] = pd.to_datetime(df_acc["timestamp"])
        df_acc["time"] = df_acc["time"].values.astype("datetime64[s]")
        df_acc["time"] = df_acc["time"].dt.tz_localize("UTC").dt.tz_convert("Europe/Berlin")
        df_acc["time"] = df_acc["time"].dt.tz_localize(None)

        df_acc = df_acc[["time", "acc_filter"]]

        fig_acc = create_fig_line(
            df_acc, "time", "acc_filter", "Movement", "Variation", "MOV", popup
        )

        if x_range is None:
            x_range = fig_acc.x_range

        fig_acc.x_range = x_range

        bokeh_pane_acc.object = fig_acc

    # HR
    if int(plot["HR"]) == 1:
        bokeh_pane_hr.visible = True
        df_hr = pd.read_csv(path_session + "/Data/df_data_hr_filtered.csv")
        df_hr["time"] = pd.to_datetime(df_hr["timestamp"])
        df_hr["time"] = df_hr["time"].values.astype("datetime64[s]")
        df_hr["time"] = df_hr["time"].dt.tz_localize("UTC").dt.tz_convert("Europe/Berlin")
        df_hr["time"] = df_hr["time"].dt.tz_localize(None)

        df_hr = df_hr[["time", "hr"]]

        fig_hr = create_fig_line(df_hr, "time", "hr", "Heart Rate", "BPM", "HR", popup)
        if x_range is None:
            x_range = fig_hr.x_range

        fig_hr.x_range = x_range

        bokeh_pane_hr.object = fig_hr

    progress_bar.visible = False


def prepare_sessions(event):
    # This method obtains the day and session from the value of the select
    global progress_bar
    progress_bar.visible = True

    global select
    groups = select.groups
    session = select.value

    day = None

    # Get the day from the string "Session #: HH:MM:SS"
    for key, values in groups.items():
        if str(session) in values:
            day = key
            break

    global path_sessions
    path_sessions = path_days + "/" + day

    global sessions
    sessions = os.listdir(path_sessions)

    # Session example: 'Session 2: 12:13:49'
    num_session = int(re.search(r"\d+", session).group())

    global current_session
    current_session = num_session_to_timestamp(num_session)

    global text_title_day, text_title_student
    text_title_day.value = "Day: " + day
    text_title_session.value = session

    visualize_session(day, current_session)


def num_session_to_timestamp(num_session):
    global sessions
    sorted_list = sorted(sessions)

    return sorted_list[num_session - 1]


def create_select_sessions(event):
    offset, thresh, start_WT, end_WT = read_param_EDA()

    global button_analyse
    global dir_input_btn

    # deactivating the buttons
    dir_input_btn.disabled = True
    button_analyse.disabled = True

    # This method converts session timestamps to the string "Session #: HH:MM:SS"
    global path_days
    days = os.listdir(path_days)

    # Dictionary with key: day      value: list of strings "Session #: HH:MM:SS"
    groups = {}
    for d in days:
        sessions = os.listdir(path_days + "/" + str(d))
        # convert session timestamps to session number in the day
        dt_objects_list = [datetime.datetime.fromtimestamp(int(t)) for t in sessions]
        dt_objects_list = pd.Series(dt_objects_list)
        dt_objects_list = dt_objects_list.dt.tz_localize("UTC").dt.tz_convert("Europe/Berlin")
        dt_objects_list = dt_objects_list.dt.tz_localize(None)
        dt_objects_list = [
            datetime.datetime.strftime(t, "%H:%M:%S") for t in dt_objects_list.to_list()
        ]
        num_sessions = rankdata(sessions).astype(int)
        string_sessions = [
            "Session " + str(n) + ": " + s
            for n, s in zip(num_sessions, dt_objects_list)
        ]

        groups[d] = string_sessions

    global select
    select.groups = groups

    global text_title_student
    text_title_student.value = "Analysing " + file_name_student
    save_data_filtered(path_days, thresh, offset, start_WT, end_WT)

    # view the first session
    prepare_sessions(event)

    dir_input_btn.disabled = False
    button_analyse.disabled = False
    select.disabled = False
    button_visualize.disabled = False


#######                 #######
#######                 #######
#######     WIDGET      #######
#######                 #######
#######                 #######

# Button to confirm the student
button_analyse = pn.widgets.Button(
    name="Analyse biometrics",
    button_type="primary",
    disabled=True,
    sizing_mode="stretch_width",
)
button_analyse.on_click(create_select_sessions)

# Progress Bar
progress_bar = pn.indicators.Progress(
    name="Progress", visible=False, active=True, sizing_mode="stretch_width"
)

# Session selection
select = pn.widgets.Select(
    name="Select Session", options=sessions, disabled=True, sizing_mode="stretch_width"
)


# Button to view the session
button_visualize = pn.widgets.Button(
    name="Visualize session",
    button_type="primary",
    disabled=True,
    sizing_mode="stretch_width",
)
button_visualize.on_click(prepare_sessions)


# Template
template = pn.template.FastGridTemplate(
    title="EmoVizPhy",
    sidebar=[dir_input_btn, button_analyse, select, button_visualize, progress_bar],
    theme_toggle=False,
)


# Header
title = pn.Row(
    pn.layout.HSpacer(), text_title_student, text_title_day, text_title_session
)
template.header.append(title)

# Main
# The number of panels shown is equal to the number of signals to show. For example, if EDA is deactivated in the config file, then its panel must be removed
show_bokeh_pane = []
if int(plot["EDA"]) == 1:
    show_bokeh_pane.append(bokeh_pane_eda)
if int(plot["HR"]) == 1:
    show_bokeh_pane.append(bokeh_pane_hr)
if int(plot["ACC"]) == 1:
    show_bokeh_pane.append(bokeh_pane_acc)

size = 2
for i in range(len(show_bokeh_pane)):
    # a maximum of 12 panels can be shown
    template.main[(i * size) : (i * size) + size, :] = show_bokeh_pane[i]



MAX_SIZE_MB = 1000
PORT = 20000

app = template
pn.serve(
    app,
    port=PORT,
    websocket_max_message_size=MAX_SIZE_MB*1024*1014,
    http_server_kwargs={'max_buffer_size': MAX_SIZE_MB*1024*1014}
)
print("Reach the application at http://localhost:20000")
