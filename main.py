from src.audio import *
from src.models import apply_action_voice
from src.gui import *
from src.model_training import *
from PIL import Image
import io
from cv2 import dnn_superres
import numpy as np
import glob


if __name__ == '__main__':

    r = sr.Recognizer()

    with sr.Microphone() as source:

        window = get_gui()

        while True:

            event, values = window.read()

            if event == "Exit" or event == sg.WIN_CLOSED:
                break

            if event == 'Load':

                original_filename = values["-FILE-"]
                filename = values["-FILE-"]
                original_path = filename

                image_uploaded = False
                while not image_uploaded:

                    image = Image.open(filename)
                    image.thumbnail((128, 128))
                    bio = io.BytesIO()
                    image.save(bio, format="PNG")
                    window["-IMAGE-"].update(data=bio.getvalue(),visible=True)
                    image_uploaded = True
                    action = "all"
                    apply_action_voice(action, file_path=filename, original=original_path)

                    differences = []
                    input_image = Image.open("output/AttGAN_128_CelebA-HQ/samples_testing_2/Input.jpg")
                    image_list = sorted(glob.glob("output/AttGAN_128_CelebA-HQ/samples_testing_2/*"))

                    image_list.remove('output/AttGAN_128_CelebA-HQ/samples_testing_2/all_images.jpg')
                    image_list.remove('output/AttGAN_128_CelebA-HQ/samples_testing_2/Reconstruction.jpg')
                    image_list.remove('output/AttGAN_128_CelebA-HQ/samples_testing_2/Input.jpg')

                    for i in image_list:
                        candidate = Image.open(i)
                        errorL2 = cv2.norm(np.array(input_image), np.array(candidate), cv2.NORM_L2)
                        similarity = 1 - errorL2 / (256 * 256)
                        differences.append(similarity)

                    ind = np.argpartition(differences, 3)

                    small_attr_dict = {
                        0: "Bald",
                        1: "Bangs",
                        2: "Black_Hair",
                        3: "Blond_Hair",
                        4: "Brown_Hair",
                        5: "Bushy_Eyebrows",
                        6: "Eyeglasses",
                        7: "Male",
                        8: "Mouth_Slightly_Open",
                        9: "Mustache",
                        10: "No_Beard",
                        11: "Pale_Skin",
                        12: "Young"
                    }

                    top3 = []
                    suggestions = []
                    for i in ind[:3]:
                        top3.append(differences[i])
                        suggestions.append(small_attr_dict[i])

                    suggestions_list = []
                    for i in suggestions:

                        if len(i.split("_")) == 3:
                            s2 = f"I can give you a beautiful smile!"
                            suggestions_list.append(s2)

                        if len(i.split("_")) > 1 and (i.split("_")[1].lower() == 'hair'
                                                      or i.split("_")[1].lower() == 'skin'
                                                      or i.split("_")[1].lower() == 'eyebrows'):

                            s1 = f"I can make your {i.split('_')[1].lower()} {i.split('_')[0].lower()}"
                            suggestions_list.append(s1)

                        if len(i.split("_")) == 1 and (i.split("_")[0].lower() != 'bangs'
                                                       and i.split("_")[0].lower() != 'mustache'
                                                       and i.split("_")[0].lower() != 'eyeglasses'):
                            s2 = f"I can make you {i.split('_')[0].lower()}"
                            suggestions_list.append(s2)

                        if len(i.split("_")) == 1 and (i.split("_")[0].lower() == 'bangs'
                                                       or i.split("_")[0].lower() == 'eyeglasses'):
                            s2 = f"I can give you {i.split('_')[0].lower()}"
                            suggestions_list.append(s2)

                    sg.popup_ok('Suggestions',
                                f"{suggestions_list[0]}\n{suggestions_list[1]}\n{suggestions_list[2]}")

                    window['ButtonKey'].click()
                    event, values = window.read()
                    window.refresh()
                    break

            if event == 'ButtonKey':

                intended_action = " "
                while event != "Exit" or event != sg.WIN_CLOSED:

                    intent_model_directory = training_intent_classification()
                    intended_action = get_action(model_directory=intent_model_directory)
                    further_intended_action = intended_action # to initialize
                    print("Your intended_action is:", intended_action)
                    window['ButtonKey2'].click()
                    event, values = window.read()
                    window.refresh()
                    break

            while event == 'ButtonKey2':

                if intended_action == further_intended_action: # first step
                    filename = values["-FILE-"]
                    apply_action_voice(intended_action, file_path=filename, original=original_path)
                else: # further steps
                    filename = 'output/AttGAN_128_CelebA-HQ/temp_images/tmp.jpg'
                    apply_action_voice(further_intended_action, file_path=filename, original=original_path)

                split = intended_action.split(" ")

                #This part is ok for 13 features, but an NLU model is necessary to make it more generic.

                for i in split:
                    if i == 'black' or i == 'dark':
                        result = Image.open("./output/AttGAN_128_CelebA-HQ/samples_testing_2/Black_Hair.jpg")
                    elif i == 'blonde' or i == 'yellow' or i=='blond':
                        result = Image.open("./output/AttGAN_128_CelebA-HQ/samples_testing_2/Blond_Hair.jpg")
                    elif i == 'brown' or i == 'brunette' or i == 'chocolate':
                        result = Image.open("./output/AttGAN_128_CelebA-HQ/samples_testing_2/Brown_Hair.jpg")
                    if i == 'bangs':
                        result = Image.open("./output/AttGAN_128_CelebA-HQ/samples_testing_2/Bangs.jpg")
                    elif i == 'pale' or i == 'light' or i == 'lighter' or i == 'white':
                        result = Image.open("./output/AttGAN_128_CelebA-HQ/samples_testing_2/Pale_Skin.jpg")
                    elif i == 'bald':
                        result = Image.open("./output/AttGAN_128_CelebA-HQ/samples_testing_2/Bald.jpg")
                    elif i == 'eyebrows' or i == 'bushy':
                        result = Image.open("./output/AttGAN_128_CelebA-HQ/samples_testing_2/Bushy_Eyebrows.jpg")
                    elif i == 'eyeglasses' or i == 'glasses':
                        result = Image.open("./output/AttGAN_128_CelebA-HQ/samples_testing_2/Eyeglasses.jpg")
                    elif i == 'close' or i == 'teeth' or i == 'tooth':
                        result = Image.open("./output/AttGAN_128_CelebA-HQ/samples_testing_2/Mouth_Slightly_Open.jpg")
                    elif i == 'wrinkles' or i == 'young':
                        result = Image.open("./output/AttGAN_128_CelebA-HQ/samples_testing_2/Young.jpg")
                    elif i == 'mustache':
                        result = Image.open("./output/AttGAN_128_CelebA-HQ/samples_testing_2/Mustache.jpg")
                    elif i == 'beard':
                        result = Image.open("./output/AttGAN_128_CelebA-HQ/samples_testing_2/No_Beard.jpg")

                try:
                    sr = dnn_superres.DnnSuperResImpl_create()
                    path = "resolution/FSRCNN-small_x3.pb"
                    sr.readModel(path)
                    sr.setModel("fsrcnn", 3)
                    result_fin = sr.upsample(np.array(result))
                    result_fin_v2 = Image.fromarray(result_fin, 'RGB')

                    filename = 'output/AttGAN_128_CelebA-HQ/temp_images/tmp.jpg'
                    with open(filename, 'w') as f:
                        result_fin_v2.save(f)

                    #bio = io.BytesIO()
                    #result_fin_v2.save(bio, format="PNG")
                    result_fin_v2.thumbnail((128, 128))
                    bio = io.BytesIO()
                    result_fin_v2.save(bio, format="PNG")
                    window["-IMAGE-"].update(data=bio.getvalue(),visible=True)
                    window['ButtonKey2'].click()
                    event, values = window.read()
                    window.refresh()

                except:
                    bio = io.BytesIO()
                    result.save(bio, format="PNG")
                    result.thumbnail((128, 128))
                    window["-IMAGE-"].update(data=bio.getvalue(), visible=True)

                    filename = 'output/AttGAN_128_CelebA-HQ/temp_images/tmp.jpg'
                    with open(filename, 'w') as f:
                        result.save(f)

                    window['ButtonKey2'].click()
                    event, values = window.read()
                    window.refresh()

                event2, _ = window.read()
                window.refresh()

                if event2 == 'Save':
                    filename = 'output/AttGAN_128_CelebA-HQ/temp_images/final_image_IRA.jpg'
                    result_fin.resize((128, 128))
                    with open(filename, 'w') as f:
                        result_fin_v2.save(f)

                    event, values = window.read()
                    window.refresh()
                    break

                elif event2 == 'Clear':
                    image = Image.open(original_filename)
                    image.thumbnail((128, 128))
                    bio = io.BytesIO()
                    image.save(bio, format="PNG")
                    window["-IMAGE-"].update(data=bio.getvalue(), visible=True)
                    window.refresh()

                    intended_action = get_action(model_directory=intent_model_directory)
                    further_intended_action = 'change_hair_color'

                    window['ButtonKey2'].click()
                    event, values = window.read()
                    window.refresh()

                elif event2 == 'Exit' or event2 == sg.WIN_CLOSED:
                    event, values = window.read()
                    window.refresh()
                    break

                elif event2 == 'Go on':
                    intended_action = get_action(model_directory=intent_model_directory)
                    window['ButtonKey2'].click()
                    event, values = window.read()
                    window.refresh()

        window.close()