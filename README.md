# Wake word detection

Wake word is the phrase that normally consists of two or more words. It is used to engage always listening AI into a full power processing. Some of the well know wake words are ‘OK Google’, ‘Hey Alexa’ or ‘Hey Siri’.

There are few open source toolkits available that can be used to build Wake Word Detection System. 

In this project we will design a machine learning model to wake from a custom wake word like ‘Hey Fourth Brain’

### Data preparation 
* We used MCV corpus 8 en https://commonvoice.mozilla.org/en/datasets - 70Gb, 80K voices, MP3 data as a source for positive (hey, fourth & brain) and negative wav files.  
* Used MFA to create timestamps for each of the MP3 files. The timestamps are needed to extract the wake words from the audio files later.
* To address the imbalanced data we used google cloud api to generate additional positive wav files.
* Recorded 200 hundred short audio files with background noise. 
* Final positive data had about 5K audio files. 
* We limited negative data to 5K audio files as well.  
### Data processing 
* Created dataloader class to handle batch data processing
* Applied MEL Spectogram to convert data suitable for audio classification.
* Make audio length similar between positive and negative samples. 
* Finally added the noise.
### Modeling approach 
* Used CNN to predict label for each of the wake words
* Design # 1: 4 conv blocks / flatten / linear / softmax => Didn’t work
* Design # 2: Conv2d-1/ReLu/MaxPool/BatchNorm/Conv2d-2/ReLu/MaxPool/BatchNorm/Linear/ReLU/Dropout/Linear => OK with some issues
* Running the model on test data produced Accuracy of 0.96 with F1 abouve 0.93 for wake words.                
       precision    recall  f1-score   support

       brain       0.99      0.95      0.97       604
      fourth       0.99      0.88      0.93       337
         hey       0.97      0.99      0.98       837
    negative       0.78      0.99      0.87       162

    accuracy                           0.96      1940
   macro avg       0.93      0.95      0.94      1940
weighted avg       0.96      0.96      0.96      1940 

### Testing
* We used javascript to record the audio in Colab. The recording was split into words and put thru the CNN prediction. 
* Even though we had pretty good accuracy and F1 score during training and testing the actual detection is proved to be far from perfect. One possible reason is the small postive sample. 

### FastAPI
* We used a simple FastAPI set up to test the deployement. 
