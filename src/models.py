from src.model_training import *
import glob


# Using the related AttGAN model to alter the image
def get_model(mode,intended_action):
    # Get the AttGAN Model
    models = []
    for name in glob.glob('/trained_models/*'):
        models.append(name)

    if mode == 'gan':
        action = intended_action
        model = "read model"
        return model
    else:
        print("You need a GAN model!")


def apply_action_voice(intended_action):
    model = get_model(mode='gan', intended_action=intended_action)
    # TODO: egitilmis attGAN'ı alacak ve fotoyu içine koyacak, sonucu cıkaracak
    a = "nothing"
    return a