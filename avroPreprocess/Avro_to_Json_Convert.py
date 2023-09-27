from avro.datafile import DataFileReader
from avro.io import DatumReader
import json

# Crea un lettore Avro
def convert_avro_to_json(file_path):
    with open(file_path, 'rb') as avro_file:
        reader = DataFileReader(avro_file, DatumReader())

        # Inizializza una lista vuota per contenere i record JSON
        json_records = []

        # Leggi e converte i dati Avro in JSON
        for record in reader:
            json_record = {}
            for key, value in record.items():
                # Puoi personalizzare la conversione in base alle tue esigenze
                # Ad esempio, puoi gestire la conversione di tipi Avro specifici
                # in tipi JSON appropriati qui
                json_record[key] = value
            json_records.append(json_record)

    # Chiudi il lettore Avro
    reader.close()

    # Scrivi i dati JSON in un file
    with open('output.json', 'w') as json_file:
        json.dump(json_records, json_file, indent=2)

    print("Conversione da Avro a JSON completata. I dati sono stati scritti in 'output.json'.")