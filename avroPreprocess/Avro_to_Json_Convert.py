from avro.datafile import DataFileReader
from avro.io import DatumReader
import json

def convert_avro_to_json(file_path):
    with open(file_path, 'rb') as avro_file:
        reader = DataFileReader(avro_file, DatumReader())
        json_records = []

        for record in reader:
            json_record = {}
            # through each key-value pair in the Avro record
            for key, value in record.items():
                # copy key-value pairs to the JSON record dictionary
                json_record[key] = value
            json_records.append(json_record)

    reader.close()

    # writing json data in file
    with open('output.json', 'w') as json_file:
        json.dump(json_records, json_file, indent=2)