import argparse
import uuid
import adafruit_dht
import redis
import sounddevice as sd
import string

import numpy as np
import torch
import torchaudio.transforms as T

from board import D4
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from scipy.io.wavfile import write
from time import time
from datetime import datetime


# Arguments
parser = argparse.ArgumentParser(description = 'Hygrometer Data Logger')
parser.add_argument('-h','--host', type=str, 
                    help='Host address for the connection to Redis Cloud database',
                    default='localhost')
parser.add_argument('-p','--port', type=int, 
                    help='Port for the connection to Redis Cloud database')
parser.add_argument('-u','--user', type=str, 
                    help='Redis Cloud username')
parser.add_argument('-pwd','--password', type=str, 
                    help='Redis Cloud password')
parser.add_argument('-v', '--verbose', action='store_true',default = False)
args = parser.parse_args()

# System Initilization
mac_address = hex(uuid.getnode())
dht_device = adafruit_dht.DHT11(D4)

redis_client = redis.Redis(
    host = args.host,
    port = args.port,
    username = args.username,
    password = args.password
)

is_connected = redis_client.ping()
print(f'Redis Connected: {is_connected}')

if not is_connected:
    exit("Cannot connect to Redis Cloud database.")

key_temp = f"{mac_address}:temperature"
key_hum = f"{mac_address}:humidity"

try:
    redis_client.ts().create(key_temp)
except redis.ResponseError:
    pass
try:
    redis_client.ts().create(key_hum)
except redis.ResponseError:
    pass

model_name = 'openai/whisper-tiny.en'
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

system_state = False

# Audio acquisition, command recognition and control logic
def callback(indata,frames,callback_time,status):
    global system_state

    indata = indata.T   #channels last --> channel first
    indata = indata.astype(np.float32)/32768.0  #indata/(2^(16Bit - 1))
    resampler = T.Resample(orig_freq=48000,new_freq=16000)
    indata = resampler(indata)
    indata = np.squeeze(indata) #remove channel dimension
    input_audio = torch.from_numpy(indata).to(torch.float32)

    input_features = processor(input_audio,
                                sampling_rate = 16000,
                                return_tensors = 'pt').input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids,
                                           skip_special_tokens=False)
    cmd = transcription[0].strip().translate(str.maketrans('', '', string.punctuation)).lower()
    print(f'Recognized command: {cmd}')

    if cmd == 'up':
        system_state = True
    elif cmd == 'stop':
        system_state = False

channels = 1
dtype = 'int16'
samplerate = 48000
blocksize = 48000  #1 second

with sd.InputStream(channels=channels,
                    dtype=dtype,
                    samplerate=samplerate,
                    blocksize=blocksize,
                    callback=callback):
    # Data collection and upload
    while True:
        if system_state == True:
            try:
                timestamp = time()
                temperature = dht_device.temperature
                humidity = dht_device.humidity

                timestamp = int(timestamp * 1000) #millisecond
                redis_client.ts().add(key_temp, timestamp, temperature)
                redis_client.ts().add('humidity', timestamp, humidity)

                time.sleep(5)
            except:
                dht_device.exit()
                dht_device = adafruit_dht.DHT11(D4)