import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import pygame
import streamlit as st

# Initialize recognizer and translator
recognizer = sr.Recognizer()
translator = Translator()

# Function to capture and translate speech
def capture_and_translate(source_lang, target_lang):
    with sr.Microphone() as source:
        st.info("🎙️ Listening... Speak now.")
        
        recognizer.adjust_for_ambient_noise(source, duration=1)
        recognizer.energy_threshold = 200

        try:
            # Capture speech
            audio = recognizer.listen(source, timeout=15, phrase_time_limit=15)
            st.success("🔄 Processing...")

            # Recognize speech
            text = recognizer.recognize_google(audio, language=source_lang)
            st.write(f"🗣️ Original ({source_lang}): {text}")

            # Translate speech
            translation = translator.translate(text, src=source_lang, dest=target_lang)
            st.write(f"🔊 Translated ({target_lang}): {translation.text}")

            # Convert translation to speech
            tts = gTTS(text=translation.text, lang=target_lang)
            audio_file = "translated_audio.mp3"
            tts.save(audio_file)

            # Play the audio
            pygame.mixer.init()
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()

            st.audio(audio_file)

            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            pygame.mixer.music.stop()
            pygame.mixer.quit()

            return audio_file

        except sr.WaitTimeoutError:
            st.error("⚠️ No speech detected within the time limit. Please ensure you're speaking into the microphone clearly and check your microphone settings.")
        except sr.UnknownValueError:
            st.error("⚠️ Speech was unclear or not recognized. Try speaking louder, clearer, or checking your microphone setup.")
        except sr.RequestError:
            st.error("⚠️ Could not request results from the speech recognition service. Please check your internet connection or try again later.")
        except Exception as e:
            st.error(f"⚠️ An unexpected error occurred: {str(e)}")
        return None
