import os
import streamlit as st
from collections import Counter

# import namespaces
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
import azure.cognitiveservices.speech as speech_sdk
global speech_config

# Get Configuration Settings
ai_key = "b32f134094a2432fa1293380952bfa61"
ai_region = "eastus"
speech_config = speech_sdk.SpeechConfig(ai_key, ai_region)
speech_config.speech_synthesis_voice_name = "en-GB-RyanNeural"
speech_synthesizer = speech_sdk.SpeechSynthesizer(speech_config)


def speak_fun(texxt):
    speak = speech_synthesizer.speak_text_async(texxt).get()
    if speak.reason != speech_sdk.ResultReason.SynthesizingAudioCompleted:
        print(speak.reason)
        
def seprate_methon_analysis(text,data_pr):
    #Loading Credentials    
    ai_endpoint = "https://lang097867575.cognitiveservices.azure.com/"
    ai_key = "d59c070ceefa417687e0b85ddf37a7c8"
    
    # Create client using endpoint and key
    credential = AzureKeyCredential(ai_key)
    ai_client = TextAnalyticsClient(endpoint=ai_endpoint, credential=credential)
    
    data_pr.text("_--:: Your text is processed , I am Telling about tour review ::---")
    # Get language
    detectedLanguage = ai_client.detect_language(documents=[text])[0]
    st.header("Language")
    st.subheader('\n{}'.format(detectedLanguage.primary_language.name),divider='rainbow')
    speak_fun(f"Language of this review is {detectedLanguage.primary_language.name}")
    
    # Get sentiment
    sentimentAnalysis = ai_client.analyze_sentiment(documents=[text])[0]
    st.header("Sentiment")
    st.subheader("\n{}".format(sentimentAnalysis.sentiment),divider='rainbow')
    
    speak_fun(f"Sentiment of this Review is {sentimentAnalysis.sentiment}")
     
     
    # Get key phrases
    phrases = ai_client.extract_key_phrases(documents=[text])[0].key_phrases
    #count_kpharas=0
    
    if len(phrases) > 5:
        with st.expander("Key Phrases (Click to expand)",expanded=True):
            st.markdown(f"Telling the first top 5 Phrases out of {len(phrases)}")
            #speak_fun("Telling the First things : Which is Key Pharases")
            speak_fun(f"Telling the first top 5 Phrases out of {len(phrases)}")
            
            st.table({"Key Phrases": phrases[:5]})
            
            
            for phrase in phrases[:5]:
                speak_fun(phrase)
    else:
        st.subheader("No KeyPhrases present in this Review ",divider='rainbow')
             
             
    # Get entities
    entities = ai_client.recognize_entities(documents=[text])[0].entities
    # st.markdown(f"{type(entities)}")
    # st.subheader(entities)
    if len(entities) > 5:
        with st.expander("Entities (Click to collapse)",expanded=True):
            st.markdown(f"Telling the first top 5 Entities out of {len(entities)}")
            speak_fun(f"Telling the first top 5 Entities out of {len(entities)}")
            entity_categories = [entity['category'] for entity in entities]

            #counting occurrences of each category
            category_counts = Counter(entity_categories)

            #top 5 categories
            top_categories = category_counts.most_common(5)

            #preparing data for the table
            table_data = [{"Category": cat, "Count": count} for cat, count in top_categories]

            #streamlit app
            st.title("Top 5 Entity Categories")
            st.table(table_data)
            
            
            #st.subheader("Speaking Top Categories")
            
            for category, count in top_categories:
                speak_fun(f"The category {category} appears {count} times.")
               
    else:
        st.subheader("No Entities present in this Review ",divider='rainbow')


    entities = ai_client.recognize_linked_entities(documents=[text])[0].entities

    if entities:
        with st.expander("Entity Links (Click to Collapse)", expanded=True):
            speak_fun("The Third thing: Which is the Links")
            linked_name = []
            linked = []
            
            #iterate through the first 5 entities
            for linked_entity in entities[:5]: 
                linked_name.append(linked_entity.name)
                linked.append(linked_entity.url)
            
            #table
            st.table({"Entity": linked_name, "Links": linked})
    else:
        st.write("No linked entities found.")

    st.header("..... Data Processed Completely ! Thanks for using  .....")


def text_analysis(flag=0,content=''):
    try:
        
        if flag==0:
            # Analyze each text file in the reviews folder
            reviews_folder = 'speech_text'
            filelist=os.listdir(reviews_folder)
            file_name=filelist[-1]
            print(file_name)
            
            #st.subheader('\nNumber of Promot asked is : ' +f"{len(os.listdir(reviews_folder))}" ,divider=True)
            text = open(os.path.join(reviews_folder, file_name), encoding='utf8').read()
            data_process=st.text("Processing your Review")
            seprate_methon_analysis(text,data_process)
   
   
        elif flag==1:
            text=content
            # st.subheader('\nNumber of Promot asked is : ' +f"{len(os.listdir(reviews_folder))}" ,divider=True)
            data_process=st.text("Processing your Review")
            seprate_methon_analysis(text,data_process)
        
        
        elif flag==2:
            text=content
            # st.subheader('\nNumber of Promot asked is : ' +f"{len(os.listdir(reviews_folder))}" ,divider=True)
            data_process=st.text("Processing your Review")
            seprate_methon_analysis(text,data_process)
        
        

        
        
        
        
        
        
        

    except Exception as ex:
        st.warning(ex)


