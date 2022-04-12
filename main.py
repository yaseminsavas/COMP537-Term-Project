from src.audio import *
from src.models import apply_action_voice
from src.model_training import *
from src.gui import *
from PIL import Image
import io
import os

if __name__ == '__main__':

    r = sr.Recognizer()

    with sr.Microphone() as source:

        window = get_gui()

        while True:

            event, values = window.read()

            if event == "Exit" or event == sg.WIN_CLOSED:
                break

            if event == 'Load':
                try:
                    filename = values["-FILE-"]

                    if os.path.exists(filename):
                        image = Image.open(filename)
                        image.thumbnail((600, 700))
                        bio = io.BytesIO()
                        image.save(bio, format="PNG")
                        window["-IMAGE-"].update(data=bio.getvalue())

                    if len(os.listdir('./rasa_project/models/')) > 0:

                        intent_model_directory = training_intent_classification()
                        intended_action = get_action(model_directory=intent_model_directory)
                        print("Your intended_action is:", intended_action)

                        # TODO: GAN MODEL TRAINING WITH FEATURES AS A PARAMETER
                        result = apply_action_voice(intended_action)
                        result.thumbnail((600, 700))
                        bio = io.BytesIO()
                        result.save(bio, format="PNG")
                        window["-IMAGE-"].update(data=bio.getvalue())

                        if event == 'Save':
                            result.save('saved_image.jpg')
                except:
                    pass

        window.close()
