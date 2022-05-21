import speech_recognition as sr
import os
from gtts import gTTS
from rasa.core.agent import Agent
import asyncio
import tensorflow as tf

# Getting the action that the user wants to apply
def get_model(model_directory):
    agent = Agent.load(model_path=model_directory)
    return agent

def get_action(model_directory):

    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = gTTS(text='Say your action out loud please', lang='en')
        audio.save("voices/intended_action.mp3")
        os.system('afplay "voices/intended_action.mp3"')
        audio = r.listen(source)

    try:
        intended_action = r.recognize_google(audio)
        #audio = gTTS(text='You said'+intended_action, lang='en')
        #audio.save("voices/message.mp3")
        #os.system('afplay "voices/message.mp3"')

        #agent = Agent.load(model_path=model_directory)
        #result = asyncio.run(agent.parse_message(message_data=intended_action))
        #tf.compat.v1.reset_default_graph()
        #return result['intent']['name'], intended_action
        return intended_action

    except sr.UnknownValueError:
        audio = gTTS(text='I could not understand what you said', lang='en')
        audio.save("voices/unknown_value_error.mp3")
        os.system('afplay "voices/unknown_value_error.mp3"')

    except sr.RequestError:
        audio = gTTS(text='I could not request the results', lang='en')
        audio.save("voices/request_error.mp3")
        os.system('afplay "voices/request_error.mp3"')