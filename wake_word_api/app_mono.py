# 1. Library imports
# import uvicorn
# import multipart
from os import spawnlp
import aiofiles
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
#from Model import CNN
from starlette.responses import FileResponse 
from fastapi import Request

from pydub import AudioSegment
from pydub.silence import split_on_silence
import matplotlib.pyplot as plt
import librosa

import ipywidgets as widgets
from IPython import display as disp
from IPython.display import display, Audio, clear_output
# from google.colab import output
import base64
from pydub import AudioSegment
import io
import tempfile
import numpy as np


import torch
import torchaudio
import sounddevice as sd
from scipy.io.wavfile import write
from torch import nn
# from torchsummary import summary

# 2. Create app and model objects
app   = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/templates", StaticFiles(directory="templates", html=True), name="templates")

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index_mono.html", {'request': request})

#origins = ['http://127.0.0.1:8000']
origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



WAKE_WORDS = ["hey", "fourth", "brain"]
class CNN(nn.Module):
    def __init__(self, num_labels, num_maps1, num_maps2, num_hidden_input, hidden_size):
        super(CNN, self).__init__()
        conv0 = nn.Conv2d(1, num_maps1, (8, 16), padding=(4, 0), stride=(2, 2), bias=True)
        pool = nn.MaxPool2d(2)
        conv1 = nn.Conv2d(num_maps1, num_maps2, (5, 5), padding=2, stride=(2, 1), bias=True)
        self.num_hidden_input = num_hidden_input
        self.encoder1 = nn.Sequential(conv0,
                                      nn.ReLU(),
                                      pool,
                                      nn.BatchNorm2d(num_maps1, affine=True))
        self.encoder2 = nn.Sequential(conv1,
                                      nn.ReLU(),
                                      pool,
                                      nn.BatchNorm2d(num_maps2, affine=True))
        self.output = nn.Sequential(nn.Linear(num_hidden_input, hidden_size),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(hidden_size, num_labels))

    def forward(self, input_data):
        x1 = self.encoder1(input_data)
        x2 = self.encoder2(x1)
        x = x2.view(-1, self.num_hidden_input)
        return self.output(x)

# Little helper function to play the audio
def play_audio(waveform, sample_rate):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  if num_channels == 1:
    display(Audio(waveform[0], rate=sample_rate))
  elif num_channels == 2:
    display(Audio((waveform[0], waveform[1]), rate=sample_rate))
  else:
    raise ValueError("Waveform with more than 2 channels are not supported.")

# Before we can test the data on the wake word phrase 
# let's create a function to trim or padd the data
def prepare_Stream(signal, sample_rate, num_samples):
    
    if sample_rate != num_samples:
        resampler = torchaudio.transforms.Resample(sample_rate, num_samples)
        signal = resampler(signal.cpu())
        print(f"sample_rate update")
    length_signal = signal[0].shape[0]

    if length_signal > num_samples:
        signal = signal[:, :num_samples]

    if length_signal < num_samples:
        num_missing_samples = num_samples - length_signal
        last_dim_padding = (0, num_missing_samples)

        signal = torch.nn.functional.pad(signal, last_dim_padding)

    return signal

# The function will load the file, split into words and then make a prediction
# If all three words are detected it will produce a note that wake word is detected
def predict_wake_word(recording_path):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    WAKE_WORDS = ["hey", "fourth", "brain"]
    num_labels = len(WAKE_WORDS) + 1 # oov
    num_maps1  = 48
    num_maps2  = 128
    num_hidden_input =  1024
    hidden_size = 128
    SAMPLE_RATE = 16000

    cnn2 = CNN(num_labels, num_maps1, num_maps2, num_hidden_input, hidden_size)
    state_dict = torch.load("wakeworddetaction_cnn7.pth", map_location=torch.device('cpu'))
    cnn2.load_state_dict(state_dict)

    # load urban sound dataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    #mel_spectrogram.to(device)

    cnn2.eval()
    
    classes = WAKE_WORDS[:]
    # negative
    classes.append("negative")

    audio_float_size = 32767

    CHUNK = 500
    CHANNELS = 1
    RATE = SAMPLE_RATE
    RECORD_MILLI_SECONDS = 1000


    testFile = recording_path

    waveform, sample_rate = torchaudio.load(testFile)
    print(f"Recording SR: {sample_rate}")
    sounddata = librosa.core.load(testFile, sr=RATE, mono=True)[0]
    print(f"Original audio")
    play_audio(waveform, sample_rate)

    sound = AudioSegment.from_wav(testFile)
    chunks = split_on_silence(sound, 
        # must be silent for at least half a second
        min_silence_len=100,

        # consider it silent if quieter than -16 dBFS
        silence_thresh=-40
    )

    
    paths = []
    for i, chunk in enumerate(chunks):
        chunk.export("./vab/chunk{0}.wav".format(i), format="wav")
        paths.append("./vab/chunk{0}.wav".format(i))

    inference_track = []
    target_state = 0
    
    for path in paths: 
    
        waveform = ''
        waveform, sample_rate = torchaudio.load(path)
        print(f"path: {path}")
        print(f"sample_rate: {sample_rate}")
        #play_audio(waveform, sample_rate)

        signal = prepare_Stream(waveform, sample_rate, 16000)

        with torch.no_grad():

            mel_audio_data = mel_spectrogram(signal.to(device)).float()

            print(f"mel_audio_data shape: {mel_audio_data.shape}")  

            predictions = cnn2(mel_audio_data.unsqueeze_(0).to('cpu'))
            print(f"predictions: {predictions}")
            predicted_index = predictions[0].argmax(0)
            print(f"predicted_index: {predicted_index}")
            predicted = classes[predicted_index]
            print(f"predicted: {predicted}")
            print(f"target_state1: {target_state}")
            label = WAKE_WORDS[target_state]
            print(f"label: {label}")

        if predicted == label:
            target_state = target_state + 1 # go to next label
            inference_track.append(predicted)
            
            print(f"target_state2: {target_state}")
            print(f"inference track: {inference_track}")
            if inference_track == WAKE_WORDS:
                print(f"Wake word {' '.join(inference_track)} detected")
                # target_state = 0
                # inference_track = []
                return(f"Wake word {' '.join(inference_track)} detected")
        elif target_state == 2: 
            target_state = 0
    return(f"Wake word is not detected: {' '.join(inference_track)}")

@app.get("/health")
def health():
    return "Service is running."


@app.post("/save", response_class=PlainTextResponse)
async def create_upload_file(file: UploadFile=File(...)):
    
    # print("filename = ", file.filename) # getting filename
    destination_file_path = "vab/FourthBrain.wav" #+ file.filename
    # print("filepath = ", destination_file_path)

    async with aiofiles.open(destination_file_path, 'wb') as out_file:
        while content := await file.read(1024):  # async read file chunk
            await out_file.write(content)  # async write file chunk
    
    return (predict_wake_word(destination_file_path))


# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)  
