## FastAPI to test wake word detection 'Hey Fourth Brain'
* You will need to run the following in your terminal before testing
    pip install fastapi
    pip install "uvicorn[standard]"
* Once you dowload the files from wake_word_api folder you are ready to test
* To start the testing first navigate to the folder with main.py and run the 'uvicorn main:app --reload' from your terminal
* Open index.html, press the button to start recording for 3 seconds. 
* If the wake word is detected you will see a corresponding phrase 'Wake word 'hey fourth brain' detected'. 
