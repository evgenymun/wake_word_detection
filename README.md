# Wake word detection

Wake word is the phrase that normally consists of two or more words. It is used to engage always listening AI into a full power processing. Some of the well known wake words are ‘OK Google’, ‘Hey Alexa’ or ‘Hey Siri’.

There are few open source toolkits available that can be used to build Wake Word Detection System. One of the great examples is [Howl](https://github.com/castorini/howl).

The ultimate goal would be creating an API that can output a model trained on user provided wake word phrase. We are not there yet. In this project we will design a machine learning model to wake from a custom wake word like ‘Hey Fourth Brain’.  

The detailed step by step approach could be found in the [wakeWordDetectioin.ipynb](https://github.com/evgenymun/wake_word_detection/blob/main/wakeWordDetection.ipynb).


### Data preparation 
* We used MCV corpus 8 en https://commonvoice.mozilla.org/en/datasets - 70Gb, 80K voices, MP3 data as a source for positive (hey, fourth & brain) and negative wav files.  
* Used MFA to create timestamps for each of the MP3 files. The timestamps are needed to extract the wake words from the audio files later.
* To address the imbalanced data we used google cloud api to generate additional positive wav files.
* Recorded 200 hundred short audio files with background noise. 
* Final positive data had about 6K audio files. 
* We limited negative data to 4K audio files as well.  
### Data structure 
* Due to the data size we didn't create a folder sturcture in github. 
* Most of the data processing on prepared in the previos step data happened in Google Colab. 
* Here is the outline of how we stored the data (a snapshot from the [wakeWordDetectioin.ipynb](https://github.com/evgenymun/wake_word_detection/blob/main/wakeWordDetection.ipynb))

./wake_words/ds

<img width="515" alt="image" src="https://user-images.githubusercontent.com/13990748/169617993-51642368-c6cd-4183-8dbc-6782967e57e6.png">


### Data processing 
* Created dataloader class to handle batch data processing
* Applied MEL Spectogram to convert data suitable for audio classification.
* Make audio length similar between positive and negative samples. 
* Finally added the noise.
### Modeling approach 
* Used CNN to predict label for each of the wake words
* Design # 1: 4 conv blocks / flatten / linear / softmax => Didn’t work
* Design # 2: Conv2d-1/ReLu/MaxPool/BatchNorm/Conv2d-2/ReLu/MaxPool/BatchNorm/Linear/ReLU/Dropout/Linear => OK with some issues
<img width="800" alt="image" src="https://user-images.githubusercontent.com/13990748/169634158-1aa47438-f108-407f-8f0b-e7fc28d73de1.png">

* Running the model on test data produced Accuracy of 0.96 with F1 abouve 0.93 for wake words.  
<img width="600" alt="image" src="https://user-images.githubusercontent.com/13990748/169634206-e1be8efd-c2fb-44b2-86c8-94cccd8e7e6b.png">


### Testing
* We used javascript to record the audio in Colab. The recording was split into words and put thru the CNN prediction.
<img width="376" alt="image" src="https://user-images.githubusercontent.com/13990748/168439167-384d640f-b3c6-490b-9b80-710461d2532d.png">
<img width="407" alt="image" src="https://user-images.githubusercontent.com/13990748/168439211-0ae8b35e-fa48-4b27-8d24-255bd192f0b7.png"> 

* Even though we had pretty good accuracy and F1 score during training and testing the actual detection is proved to be far from perfect. One possible reason is the small postive sample. 

### FastAPI
* We used FastAPI to test the functionality before the deployement (see [wake_word_api](https://github.com/evgenymun/wake_word_detection/tree/main/wake_word_api))
* API consists of the front end (HTML, .js) and backend Fastapi, Uvicorn which we later hosted on AWS EC2 Ubuntu instance

### Deployement
* We deployed the API on the AWS EC2 port 5000 for SSL connection 
* SSL was required for audio recording access in the browser
* Deployment was new to all of the members but with great support from Camilo we were able to get it done  
* To use uvicorn with ssl we generated the key and chain using CERTBOT
* Since we didn't have the domain name Camilo kindly pointed it to our AWS IP 
![image](https://user-images.githubusercontent.com/13990748/169619245-7247ed8b-2a0b-4d25-a04c-81b160275e60.png)



### Next steps
* The API is just in the begining stage now. 
* We will be adding better code structure 
* The detection is done on the saved file at this point. Instream wake word detection will be added next. 
* Also the file is split on the chunks that are saved on the server. Obviously if multiple people testing that presents some challenges. So the next step will be to add temporary file structure to avoid the ovewriting. 
* After the clean up process is complete the next step will be to generate 'data' as a product service where we provide training data for the user. 
* We also plan to provide an API to train your own wake word and download the pth file

### Thank you Fourth Brain team for an amazing journey into the world of MLE! 
