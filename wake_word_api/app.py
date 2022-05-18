# 1. Library imports
import uvicorn
import multipart
import aiofiles
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request

from pydub import AudioSegment
from pydub.silence import split_on_silence
import matplotlib.pyplot as plt
import librosa

import ipywidgets as widgets
from IPython import display as disp
from IPython.display import display, Audio, clear_output
import base64

from pydub import AudioSegment
# from ffprobe import FFProbe

import io
import os
import tempfile
import numpy as np


import torch
import torchaudio
import torchaudio.transforms as T
from torch import nn

from scipy.io.wavfile import write

# from torchsummary import summary

# 2. Create app and model objects
app   = FastAPI()
#model = CNN()

templates = Jinja2Templates(directory="templates")
app.mount("/templates", StaticFiles(directory="templates", html=True), name="templates")

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {'request': request})

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
        pool  = nn.MaxPool2d(2)
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



# Before we can test the data on the wake word phrase 
# let's create a function to trim or padd the data
def prepare_Stream(signal, num_samples):

    length_signal = signal[0].shape[0]

    if length_signal > num_samples:
        signal = signal[:, :num_samples]

    if length_signal < num_samples:
        num_missing_samples = num_samples - length_signal
        last_dim_padding = (0, num_missing_samples)

        signal = torch.nn.functional.pad(signal, last_dim_padding)

    return signal

def stereo_to_mono (file):
    # Open the stereo audio file as an AudioSegment instance
    stereo_audio = AudioSegment.from_file(file, format="wav")

    # Calling the split_to_mono method on the stereo audio file
    mono_audios = stereo_audio.split_to_mono()
    mono_file = mono_audios[0].export(file,    format="wav")
    pass

def resamle_file (file, resampled_rate):

    waveform, sample_rate = torchaudio.load(file)
    resampler = T.Resample(sample_rate, resampled_rate, dtype=waveform.dtype)
    resampled_waveform = resampler(waveform)

    torchaudio.save(file, resampled_waveform, resampled_rate)
    pass


# The function will load the file, split into words and then make a prediction
# If all three words are detected it will produce a note that wake word is detected
def predict_wake_word(audioFile):
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
    CHUNK = 500
    CHANNELS = 1
    RATE = SAMPLE_RATE
    RECORD_MILLI_SECONDS = 1000
    audio_float_size = 32767


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

    cnn2.eval()
    
    classes = WAKE_WORDS[:]
    # negative
    classes.append("negative")


    stereo_to_mono (audioFile)
    resamle_file (audioFile, RATE)


    waveform, sample_rate = torchaudio.load(audioFile)
    print(f"Recording SR: {sample_rate}")
    
    
    #splitting Audio file to chunks of words
    sound = AudioSegment.from_wav(audioFile)
    # must be silent for at least half a second
    # consider it silent if quieter than -16 dBFS
    chunks = []
    chunks = split_on_silence(sound, min_silence_len=100, silence_thresh=-40)
       
    
    paths = []
    for i, chunk in enumerate(chunks):
        print(f"Chunk number: {i}")
        chunk.export("./vab/chunk{0}.wav".format(i), format="wav")
        paths.append("./vab/chunk{0}.wav".format(i))

    inference_track = []
    target_state = 0
    
    for path in paths: 
    
        waveform = ''
        waveform, sample_rate = torchaudio.load(path)
        signal = prepare_Stream(waveform, 16000)
        #os.remove(path)

        with torch.no_grad():

            mel_audio_data = mel_spectrogram(signal.to(device)).float()

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
                #return(f"Wake word {' '.join(inference_track)} detected")
                return jsonable_encoder({   'prediction':1, 
                                            'words': {' '.join(inference_track)}})
        elif target_state == 2: 
            target_state = 0

    return jsonable_encoder({'prediction':0, 
                             'words': {' '.join(inference_track)}})
    

@app.get("/health")
def health():
    return "Service is running."


@app.post("/save")
async def create_upload_file(file: UploadFile=File(...)):
    
    print("filename = ", file.filename) # getting filename
    destination_file_path = "vab/"+ file.filename
    print("filepath = ", destination_file_path)

    async with aiofiles.open(destination_file_path, 'wb') as out_file:
        while content := await file.read(1024):  # async read file chunk
            await out_file.write(content)  # async write file chunk
    
    return (predict_wake_word(destination_file_path))
    

# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)  