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
import whisper 
from time import time, sleep

# Arguments
parser = argparse.ArgumentParser(description='Hygrometer Data Logger')
parser.add_argument('-H', '--host', type=str,
                    help='Host address for the connection to Redis Cloud database',
                    default='localhost')
parser.add_argument('-p', '--port', type=int,
                    help='Port for the connection to Redis Cloud database')
parser.add_argument('-u', '--user', type=str,
                    help='Redis Cloud username')
parser.add_argument('-pwd', '--password', type=str,
                    help='Redis Cloud password')
parser.add_argument('-v', '--verbose', action='store_true', default=False)
args = parser.parse_args()

# System Initialization
mac_address = hex(uuid.getnode())
dht_device = adafruit_dht.DHT11(D4)

redis_client = redis.Redis(
    host=args.host,
    port=args.port,
    username=args.user,
    password=args.password,
    decode_responses=False
)

is_connected = redis_client.ping()
print(f'Redis Connected: {is_connected}')

if not is_connected:
    exit("Cannot connect to Redis Cloud database.")

key_temp = f"{mac_address}:temperature"
key_hum = f"{mac_address}:humidity"

# Create TimeSeries if they don't exist
try:
    redis_client.execute_command('TS.CREATE', key_temp, 'DUPLICATE_POLICY', 'LAST')
    print(f"Created TimeSeries: {key_temp}")
except Exception as e:
    print(f"TimeSeries {key_temp}: {e}")

try:
    redis_client.execute_command('TS.CREATE', key_hum, 'DUPLICATE_POLICY', 'LAST')
    print(f"Created TimeSeries: {key_hum}")
except Exception as e:
    print(f"TimeSeries {key_hum}: {e}")

# Load whisper model directly
model = whisper.load_model("tiny")
system_state = False
resampler = T.Resample(orig_freq=48000, new_freq=16000)

def process_audio_chunk():
    """Record and process 1 second of audio for voice commands"""
    try:
        # Record 1 second of audio
        recording = sd.rec(
            int(1.0 * 48000),
            samplerate=48000,
            channels=1,
            dtype='int16'
        )
        sd.wait()
        
        audio_data = np.squeeze(recording)
        audio_data = audio_data.astype(np.float32) / 32768.0
        
        # Convert to PyTorch tensor and resample
        audio_tensor = torch.from_numpy(audio_data).to(torch.float32)
        resampled_audio = resampler(audio_tensor)
        processed_audio = resampled_audio.squeeze()

        # Process with Whisper
        result = model.transcribe(processed_audio.numpy(), fp16=False)
        cmd = result["text"].strip().translate(
            str.maketrans('', '', string.punctuation)
        ).lower()
        
        print(f'Recognized: "{cmd}"')
        
        # Use partial matching
        if 'up' in cmd:
            return True
        elif 'stop' in cmd:
            return False
        
    except Exception as e:
        print(f"Error in audio processing: {e}")
    
    return None

def store_sensor_data(temperature, humidity):
    """Store sensor data in Redis TimeSeries"""
    try:
        timestamp_ms = int(time() * 1000)  # Current time in milliseconds
        
        # Store data using Redis TimeSeries
        redis_client.execute_command('TS.ADD', key_temp, timestamp_ms, temperature)
        redis_client.execute_command('TS.ADD', key_hum, timestamp_ms, humidity)
        
        print(f"✓ Data stored - Time: {timestamp_ms}, Temp: {temperature}°C, Hum: {humidity}%")
        return True
        
    except Exception as e:
        print(f"✗ Failed to store data in TimeSeries: {e}")
        return False

print("Starting system...")
print("Say 'up' to enable data collection, 'stop' to disable")

# Main loop
while True:
    # Process voice commands every 1 second
    command_result = process_audio_chunk()
    if command_result is not None:
        system_state = command_result
        if system_state:
            print("*** SYSTEM STATE: ENABLED - Data collection STARTED ***")
        else:
            print("*** SYSTEM STATE: DISABLED - Data collection STOPPED ***")
    
    # Data collection when enabled
    if system_state:
        try:
            temperature = dht_device.temperature
            humidity = dht_device.humidity
            
            if temperature is not None and humidity is not None:
                store_sensor_data(temperature, humidity)
            else:
                print("Failed to read sensor data")
            
            sleep(5)
            
        except Exception as e:
            print(f"Sensor error: {e}")
            try:
                dht_device.exit()
            except:
                pass
            dht_device = adafruit_dht.DHT11(D4)
            sleep(1)
    else:
        sleep(1)  # Check every 1 second when disabled
