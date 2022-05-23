import subprocess

import speech_recognition as sr
import os

from gtts import gTTS
from rasa.core.agent import Agent
import asyncio
import tensorflow as tf

def get_action(model_directory):

   #while True:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio = gTTS(text='Say your action out loud please', lang='en')
            #audio.save("voices/intended_action.mp3")
            #os.system('afplay "voices/intended_action.mp3"')
            #audio = r.listen(source)
        #try:
            #intended_action = r.recognize_google(audio)
            intended_action = "yellow hair"
            print(intended_action) #burası ok
            #os.system('python rasa shell -m models/nlu-20220523-212532-endothermic-pergola.tar.gz ')
            os.system('python rasa shell -m models/nlu-20220523-212532-endothermic-pergola.tar.gz ')
            #agent = Agent.load(model_path=model_directory) #bu olmuyo
            #print(agent)
            #result = asyncio.run(agent.parse_message(intended_action))
            #print(result)
            #yield result['intent']['name'], intended_action

        #except:
        #    continue