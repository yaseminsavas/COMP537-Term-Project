import PySimpleGUI as sg
import os.path
import cv2

# Getting the image that the user wants to alter
def get_gui():

    file_types = [("JPEG (*.jpg)", "*.jpg"),
                  ("All files (*.*)", "*.*")]

    exit_col = [[sg.Button('Exit')]]

    volume_col = [[sg.Text("Volume"),
                   sg.Slider(key='volume', range=(0, 100),
                             orientation='h', size=(10, 15), default_value=100, border_width=0,
                             enable_events=True)]]

    column_to_be_centered = [
        [sg.Image(key="-IMAGE-", size=(600, 700), background_color="white")],
        [sg.FileBrowse(file_types=file_types, enable_events=True, key="-FILE-"), sg.Button("Load"), sg.Button("Camera"), sg.Button("Save")]]

    sg.theme('LightPurple')

    layout = [[sg.Column(exit_col, element_justification="right", vertical_alignment="bottom", expand_x=True, )],
              [sg.Column(volume_col, element_justification="right", vertical_alignment="top", expand_x=True, )],
              [sg.Column(column_to_be_centered, element_justification='center')]]

    window = sg.Window("IRA", layout, element_justification='c')

    return window





