import adafruit_dht
import uuid
import time
from datetime import datetime
from board import D4

import argparse

import redis

import torch
import string
from transformers import WhisperProcessor, WhisperForConditionalGeneration

import sounddevice as sd
from time import time
from scipy.io.wavfile import write


mac_address = hex(uuid.getnode())
# print("Device MAC Address:", mac_address)

#initialize the DHT11 device
dht_device = adafruit_dht.DHT11(D4)

# parameters accepted by the script (they are optional for now; to make them required, add 'required=True' in the add_argument() function)
parser = argparse.ArgumentParser(description='Hygrometer Data Logger')
parser.add_argument('-h','--host', type=str, help='Host address for the connection to Redis Cloud database') #, default='localhost')
parser.add_argument('-p','--port', type=int, help='Port for the connection to Redis Cloud database')
parser.add_argument('-u','--username', type=str, help='Redis Cloud username')
parser.add_argument('-pwd','--password', type=str, help='Redis Cloud password')
parser.add_argument('-v', '--verbose', action='store_true')  # on/off flag
args = parser.parse_args()

# connection to Redis Cloud database
redis_client = redis.Redis(
    host=args.host,
    port=args.port,
    username=args.username,
    password=args.password
)
is_connected = redis_client.ping()
print('Redis Connected:', is_connected)
if not is_connected:
    exit("Cannot connect to Redis Cloud database. Exiting. You should insert valid connection parameters: host, port, username, password.")

# create or load timeseries in Redis Cloud database
try:
    redis_client.ts().create('temperature')
except redis.ResponseError:
    pass
try:
    redis_client.ts().create('humidity')
except redis.ResponseError:
    pass

# load Whisper tiny model and processor
model_name = 'openai/whisper-tiny.en'
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# set system state to disabled initially (False if disabled, True if enabled)
state = False

# callback function to process audio input, recognize commands, and change system state. It works every second (blocksize=48000 at 48000 Hz)
def callback(indata, frames, callback_time, status):
    global state
    
    # process audio input for command recognition
    # input_audio = indata.flatten().astype('float32')
    input_audio = torch.from_numpy(indata).to(torch.float32)
    input_features = processor(
        input_audio,
        sampling_rate=16000,
        return_tensors='pt',
    ).input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(
        predicted_ids, skip_special_tokens=False
    )
    # command = transcription[0].strip().lower()
    command = transcription[0].strip().translate(str.maketrans('', '', string.punctuation)).lower()
    # print(f'Recognized command: {command}')

    if command == 'up':
        state = True
        # print('Hygrometer logging ENABLED')
    elif command == 'stop':
        state = False
        # print('Hygrometer logging DISABLED')


# the blocksize calls the callback every second (48000 samples at 48000 Hz)
with sd.InputStream(device=1, channels=1, dtype='int16', samplerate=48000, callback=callback, blocksize=48000):
    while True: 
        # send hygrometer data to Redis Cloud database only if the system state is enabled
        if state:            
            try:
                temperature = dht_device.temperature
                humidity = dht_device.humidity
                
                timestamp = time.time()  # returns Unix time in seconds
                timestamp_ms = int(timestamp * 1000)  # Convert Unix time in milliseconds and cast it to integer
                redis_client.ts().add('temperature', timestamp_ms, f"{mac_address}:{temperature}")
                redis_client.ts().add('humidity', timestamp_ms, f"{mac_address}:{humidity}")
                # here I can use madd() to send both temperature and humidity in a single command


                # store hygrometer data in Redis Cloud database (suggested by Copilot, but not used here)
                # redis_client.hset('hygrometer_data', mapping={
                #     'timestamp': formatted_date,
                #     'temperature': temperature,
                #     'humidity': humidity
                # })
            except:
                dht_device.exit()
                dht_device = adafruit_dht.DHT11(D4)
            time.sleep(5)