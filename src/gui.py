import PySimpleGUI as sg
import os.path
import cv2

# Getting the image that the user wants to alter
def get_gui():

    file_types = [("JPEG (*.jpg)", "*.jpg"),
                  ("All files (*.*)", "*.*")]

    exit_col = [[sg.Button('Exit')]]

    column_to_be_centered = [
        [sg.Image(key="-IMAGE-", size=(128, 128), background_color="white")],
        [sg.FileBrowse(file_types=file_types, enable_events=True, key="-FILE-"), sg.Button("Load"),sg.Button("Go on"), sg.Button("Clear"), sg.Button("Save")]]

    sg.theme('LightPurple')

    layout = [[sg.Column(exit_col, element_justification="right", vertical_alignment="bottom", expand_x=True, )],
              [sg.Column(column_to_be_centered, element_justification='center'),sg.Button('ButtonKey',visible=False),
               sg.Button('ButtonKey2',visible=False)]]

    window = sg.Window("IRA", layout, element_justification='c')

    return window





