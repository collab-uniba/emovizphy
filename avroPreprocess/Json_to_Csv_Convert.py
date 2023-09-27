import json
import pandas as pd
import os

def convert_json_to_csv(filename, dirname):
    with open(filename, 'r') as json_file:
        data_list = json.load(json_file)

    timestamps_eda = []
    sampling_frequencies_eda = []
    eda_values = []
    eda_timestamp = None
    eda_sampling_frequency = 4.000000
    child_folder_name = "Data"
    
    # Itera su ciascun elemento del json
    for item in data_list:
        rawData = item.get("rawData", {})   #aggiungere costante per il nome sensore che sarà dinamico
        eda_data = rawData.get("eda", {})

        # Estrai il timestamp e sampling frequency solo se non è stato già estratto
        if eda_timestamp is None:
            # Estrai il timestamp e la frequenza di campionamento dall'EDA (se disponibili)
            eda_timestamp = eda_data.get("timestampStart", None)
            eda_timestamp = int(eda_timestamp/1000000)
            #eda_sampling_frequency = eda_data.get("samplingFrequency", None)

            # Aggiungi i timestamp e le frequenze di campionamento all'elenco
            timestamps_eda.append(eda_timestamp)
            sampling_frequencies_eda.append(eda_sampling_frequency)
        
            if eda_data:
                # Estrai i dati EDA
                eda_values.extend(eda_data.get("values", []))

    # Aggiungi la cartella figlia a dirname con il separatore di percorso appropriato
    path_join = os.path.join(dirname, child_folder_name)
    if not os.path.exists(path_join):
        os.makedirs(path_join)
            
    df1 = pd.DataFrame({"Value": timestamps_eda + sampling_frequencies_eda + eda_values})
    df1.to_csv(os.path.join(os.path.join(dirname, child_folder_name), 'EDA.csv'), index=False, header=False, float_format='%.6f')
    
    print("Dati dell'accelerometro ed eda estratti e scritti nei rispettivi file")

    return dirname