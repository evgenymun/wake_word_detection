# Wake word detection

Wake word is the phrase that normally consists of two or more words. It is used to engage always listening AI into a full power processing. Some of the well know wake words are ‘OK Google’, ‘Hey Alexa’ or ‘Hey Siri’ and now 'Hey Fourth Brain'.

There are few open source toolkits available that can be used to build Wake Word Detection System. 

https://commonvoice.mozilla.org/en/datasets - 70Gb, 80K voices, MP3

In this project we will design a machine learning model to wake from a custom wake word like ‘Hey Forth Brain’

### Data preparation 
Used MCV corpus 8 en 70GB data as a source for positive (hey, fourth & brain) and negative wav files 
Used MFA to create timestamps for each of the MP3 files, needed to extract the wake words from the audio files later
To address imbalanced data we used google cloud api to generate additional positive wav files
Recorded 200 hundred short audio files with background noise 
Final positive data had about 5K audio files 
We limited negative data to 5K audio files as well 
### Data processing 
Created dataloader class to handle batch data processing
Apply MEL Spectogram to convert data suitable for audio classification
Make audio length similar between positive and negative samples 
Add noise
#### Modeling approach 
Used CNN to predict label for each of the wake words
4 conv blocks / flatten / linear / softmax => Didn’t work
Conv2d-1/ReLu/MaxPool/BatchNorm/Conv2d-2/ReLu/MaxPool/BatchNorm/Linear/ReLU/Dropout/Linear => OK with some issues
Need to think how to address overfitting 
### Testing

### FastAPI
