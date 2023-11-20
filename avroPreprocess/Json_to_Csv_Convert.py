import json
import pandas as pd
import os
from datetime import datetime

def pulse_rate_export(timestamp_start, sampling_frequency, num_values, dirname, path_pulse_rate):
    pulserate_values = []
    
    timestamp_end = timestamp_start + (num_values/sampling_frequency) # calculating end timestamp
    
    # convert timestamp to hour
    hour_start = pd.to_datetime(timestamp_start, unit='s', utc=True).tz_convert("Europe/Berlin").tz_localize(None).time()
    hour_end = pd.to_datetime(timestamp_end, unit='s', utc=True).tz_convert("Europe/Berlin").tz_localize(None).time()

    # get the pulse rate (is only one)
    pulse_rate_files = [file for file in os.listdir(path_pulse_rate) if file.endswith("pulse-rate.csv")]
    file_path = os.path.join(path_pulse_rate, pulse_rate_files[0])
    
    df = pd.read_csv(file_path)
    for index, row in df.iterrows():
        date_pulse_rate = row.iloc[1].rstrip('Z')
        hour = datetime.fromisoformat(date_pulse_rate).time() # convert the date string to a datetime object and extract the hour
        if hour_start < hour < hour_end and row.iloc[4] != "device_not_recording":  # checking if hour is in the range of the session and if is not "device_not_recording"
                pulserate_values.append(row.iloc[3]) # append the pulse rate value
    
    timestamp_hr = [timestamp_start]
    sf_hr = [sampling_frequency]
    
    df_pulse_rate = pd.DataFrame({"Value": timestamp_hr + sf_hr + pulserate_values})
    df_pulse_rate.to_csv(os.path.join(dirname, 'HR.csv'), index=False, header=False, float_format='%.6f')
        
def convert_json_to_csv(filename, dirname, path_pulse_rate):
    with open(filename, 'r') as json_file:
        data_list = json.load(json_file)

    timestamps_eda = []
    sampling_frequencies_eda = []
    eda_values = []
    eda_timestamp = None
    eda_sampling_frequency = 4.000000
    tags_values = []

    # for every element in the json
    for item in data_list:
        rawData = item.get("rawData", {})
        eda_data = rawData.get("eda", {})
        tags_data = rawData.get("tags", {}).get("tagsTimeMicros", [])
        
        # extract the timestamp and sampling frequency from the EDA data if they are not already extracted
        if eda_timestamp is None: 
            # extract the timestamp and sampling frequency from the EDA data
            eda_timestamp = eda_data.get("timestampStart", None)
            eda_timestamp = int(eda_timestamp/1000000)

            # add the timestamps and sampling frequencies to the list
            timestamps_eda.append(eda_timestamp) 
            sampling_frequencies_eda.append(eda_sampling_frequency)
        
            if eda_data:
                eda_values.extend(eda_data.get("values", [])) # extracting eda values
                
        if tags_data: # extracting tags values
            tags_values.extend([timestamp / 1000000 for timestamp in tags_data])
            df_tags = pd.DataFrame(tags_values)
            df_tags.to_csv(os.path.join(dirname, 'tags.csv'), index=False, header=False, float_format='%.2f')
            
    df1 = pd.DataFrame({"Value": timestamps_eda + sampling_frequencies_eda + eda_values})
    df1.to_csv(os.path.join(dirname, 'EDA.csv'), index=False, header=False, float_format='%.6f')
    
    number_eda_values = len(eda_values) # number of eda values necessary to calculate the timestamp end for the pulse rate
    pulse_rate_export(eda_timestamp, int(eda_sampling_frequency), number_eda_values, dirname, path_pulse_rate)