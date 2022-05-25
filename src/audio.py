import speech_recognition as sr
import os

from gtts import gTTS


def get_action():

    while True:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio = gTTS(text='Say your action out loud please', lang='en')
            audio.save("voices/intended_action.mp3")
            os.system('afplay "voices/intended_action.mp3"')
            audio = r.listen(source)
        try:
            intended_action = r.recognize_google(audio)
            print(intended_action) #burası ok
            return intended_action
        except:
            continue