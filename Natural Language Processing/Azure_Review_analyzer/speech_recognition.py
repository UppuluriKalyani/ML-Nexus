from datetime import datetime
import os
from text_sentimet import text_analysis
import streamlit as st 


# Import namespaces
import azure.cognitiveservices.speech as speech_sdk
from playsound import playsound


def main():
    try:
        global command1
        global speech_config

        # Get Configuration Settings
   
        ai_key = "b32f134094a2432fa1293380952bfa61"
        ai_region = "eastus"

        # Configure speech service
        speech_config = speech_sdk.SpeechConfig(ai_key, ai_region)
        st.subheader(f'Ready to use speech service in: {speech_config.region}',divider='rainbow')
        #print('Ready to use speech service in ', speech_config.region)
        st.divider()
        
        if st.button(" Start by giving voice input Text"):
            command1 = TranscribeCommand()
            
            drictory="speech_text"
            #maintatinig the file count in directory 
            fileslist=os.listdir("speech_text")
            lenfl=len(fileslist)
            # print(fileslist)
            # print(lenfl)
            
            if command1 != '':
                with open(f"speech_text/review_{lenfl+1}.txt","x") as f:
                    #print("int the open funtion")
                    f.write(f"{command1}") 
                    st.subheader(f"Your speaked text is saved in File name : {f"reveiw_{lenfl+1}.txt"}")
                    f.close()     
                text_analysis()
            else:
                st.warning("I cant hear any specific command kindly speak Again by Using button")

    except Exception as ex:
        st.warning(ex)

def TranscribeCommand():
    command = ''


    # Configure speech recognition
    audio_config = speech_sdk.AudioConfig(use_default_microphone=True)
    speech_recognizer = speech_sdk.SpeechRecognizer(speech_config, audio_config)
    st.subheader('Speak now...' ,divider='rainbow')


    # Process speech input
    speech = speech_recognizer.recognize_once_async().get()
    if speech.reason == speech_sdk.ResultReason.RecognizedSpeech:
        command = speech.text
        spoken_text = st.text_area("Simulated Voice Ouput",placeholder=f"{command}")
        if spoken_text:
            st.markdown(f"""
            <div style="
                background-color: #f0f0f5;
                padding: 8px;
                border-radius: 13px;
                box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
                font-size: 30px;
                font-family: Arial, sans-serif;
                color: #80ff00;
                text-align: center;
            ">
                {spoken_text}
            </div>
            """, unsafe_allow_html=True)

        
        
        
        
        
    else:
        print(speech.reason)
        if speech.reason == speech_sdk.ResultReason.Canceled:
            cancellation = speech.cancellation_details
            print(cancellation.reason)
            print(cancellation.error_details)

    # Return the command
    return command



if __name__ == "__main__":
    main()

# def TellTime():
#     now = datetime.now()
#     response_text = 'The time is {}:{:02d}'.format(now.hour,now.minute)


#     # Configure speech synthesis
#     speech_config.speech_synthesis_voice_name = "en-GB-RyanNeural"
#     speech_synthesizer = speech_sdk.SpeechSynthesizer(speech_config)

    
#     # Synthesize spoken output
#     speak = speech_synthesizer.speak_text_async(response_text).get()
#     if speak.reason != speech_sdk.ResultReason.SynthesizingAudioCompleted:
#         print(speak.reason)


#     # Print the response
#     print(response_text)

