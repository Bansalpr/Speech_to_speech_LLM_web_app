import streamlit as st
import speech_recognition as sr
import pyttsx3
from transformers import pipeline

# Initialize speech recognizer and text-to-speech engine
r = sr.Recognizer()
engine = pyttsx3.init()

# Load a pre-trained LLM (GPT-2)
generator = pipeline('text-generation', model='gpt2')

# Streamlit app
def main():
    st.title("Speech-to-Speech Bot with Hugging Face")

    # Speech input section
    st.header("Speak to the Bot")
    if st.button("Start Listening"):
        with sr.Microphone() as source:
            audio = r.listen(source)

            try:
                text = r.recognize_google(audio)
                st.write(f"You said: {text}")

                # LLM interaction
                response = generator(text, max_length=50, num_return_sequences=1)[0]['generated_text']
                st.write(f"Response: {response}")

                # Text-to-speech output (Streamlit)
                # **Use pyttsx3 to generate audio**
                engine.say(response)
                engine.runAndWait()

                # **Play the audio in Streamlit **

            except sr.UnknownValueError:
                st.error("Could not understand audio")
            except sr.RequestError as e:
                st.error(f"Could not request results from Google Speech Recognition service; {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
