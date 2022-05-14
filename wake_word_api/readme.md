## FastAPI to test wake word detection 'Hey Fourth Brain'
* You will need to run the following in your terminal before testing

    pip install fastapi
    
    pip install "uvicorn[standard]"
    
* Once you dowload the files from wake_word_api folder you are ready to test
* To start the testing run the following command from your terminal: 

    uvicorn main:app --reload
    
* Open index.html, press the button to start recording for 3 seconds.
<img width="273" alt="image" src="https://user-images.githubusercontent.com/13990748/168440616-fa501bb2-2bb3-428d-828a-c18bb61a250b.png"> 
* If the wake word is detected you will see a corresponding phrase 'Wake word 'hey fourth brain' detected'. 
