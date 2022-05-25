import os
import numpy as np
import tensorflow as tf
import cv2
import tqdm
import yaml
import src
tf.get_logger().setLevel('ERROR')


"""
These modules are from the original repository
Github Reference: https://github.com/LynnHo/AttGAN-Tensorflow
Paper Reference: https://ieeexplore.ieee.org/abstract/document/8718508
"""
import tflib as tl
import imlib as im
import src.src_AttGAN.data as data
import src.src_AttGAN.pylib.argument
import src.src_AttGAN.pylib as py
import src.src_AttGAN.module as module

def apply_action_voice(intended_action, file_path,original):

    """
    Some of these codes are taken and adapted from the original AttGAN paper.
    src_AttGAN repository contains the codes I took from the original repository.
    Part of this code is from the test.py file from the original repository.
    It contains some other functions too. src_AttGAN directory,  tflib, and imlib directories are from the original repo.

    Github Reference: https://github.com/LynnHo/AttGAN-Tensorflow
    Paper Reference: https://ieeexplore.ieee.org/abstract/document/8718508

    """

    img_dir = file_path
    txt_list = ['data/celebA-HQ/test_label.txt','data/celebA-HQ/train_label.txt','data/celebA-HQ/val_label.txt']

    test_label_path = ''

    for file in txt_list:
        with open(file, 'r') as f:
            for line in f:
                if line.startswith(original.split("/")[-1]):
                    test_label_path = file
                    break

    test_int = 2
    experiment_name = "AttGAN_128_CelebA-HQ"

    args_ = py.args()
    output_dir = py.join('output', experiment_name)

    # save settings
    args = {}
    with open(py.join(output_dir, 'settings.yml'), 'r') as stream:
        try:
            args = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    args.update(args_.__dict__)

    # others
    n_atts = len(args["att_names"])

    sess = tl.session()
    sess.__enter__()

    ATT_ID = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2,
              'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6,
              'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10,
              'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13,
              'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16,
              'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19,
              'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22,
              'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25,
              'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28,
              'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31,
              'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34,
              'Wearing_Hat': 35, 'Wearing_Lipstick': 36,
              'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}

    img_paths = original #img_dir
    s2 = img_paths.split("/")[-1]

    label = []
    if test_label_path != '':
        with open(test_label_path, 'r') as f:
            for line in f:
                if line.startswith(s2):
                    label = line[len(s2):]
                    break

    if len(label) > 0:
        a = label[2:-1].split(" ")
        a_fin = []
        for i in a:
            i = int(i)
            a_fin.append(i)

        label = np.take(a_fin, np.array([ATT_ID[att_name] for att_name in args["att_names"]]))
        label_v2 = tf.constant(label)
        label_v2 = (label_v2 + 1) // 2
        labels = label_v2[np.newaxis]

        image = tf.compat.v1.io.read_file(img_dir)
        image = tf.compat.v1.image.decode_png(image, 3)
        image_v2 = image[np.newaxis]
        image_v2 = tf.compat.v1.image.resize(image_v2, [args["load_size"], args["load_size"]])
        image_v2 = tf.compat.v1.image.resize_image_with_pad(image_v2, args["crop_size"], args["crop_size"])
        image_v2 = tl.center_crop(image_v2, size=args["crop_size"])
        fin_image = tf.compat.v1.clip_by_value(image_v2, 0, 255) / 127.5 - 1

    else:

        """ 
        To alter images more than the ones in the dataset, we need to generate labels.
        It is possible to do it via feeding the image to the encoder and getting the result.
        However, this code below has some bugs. Currently, I give the label manually for my image. 
        However, the end results are kinda creepy...
        """

        """
        image = tf.compat.v1.io.read_file(img_dir)
        image = tf.compat.v1.image.decode_png(image, 3)
        image_v2 = image[np.newaxis]
        image_v2 = tf.compat.v1.image.resize(image_v2, [args["load_size"], args["load_size"]])
        image_v2 = tl.center_crop(image_v2, size=args["crop_size"])
        fin_image = tf.compat.v1.clip_by_value(image_v2, 0, 255) / 127.5 - 1

        Genc, Gdec, _ = src.src_AttGAN.module.get_model(args["model"], n_atts, weight_decay=args["weight_decay"])
        xa = tf.compat.v1.placeholder(tf.float32, shape=[None, args["crop_size"], args["crop_size"], 3])
        
        enc = Genc(xa, training=False)
        f1 = sess.run([fin_image])
        label = sess.run(enc, feed_dict={xa: f1})
        
        label_v2 = tf.constant(label)
        label_v2 = (label_v2 + 1) // 2
        labels = label_v2[np.newaxis]
        """

        label = " -1 -1 1 -1 -1 -1 1 1 -1 -1 -1 1 -1 -1 -1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 1 -1 -1 -1 1 -1 1 1 -1 -1 -1 -1 -1 -1 1 "
        a = label[2:-1].split(" ")
        a_fin = []
        for i in a:
            i = int(i)
            a_fin.append(i)

        label = np.take(a_fin, np.array([ATT_ID[att_name] for att_name in args["att_names"]]))
        label_v2 = tf.constant(label)
        label_v2 = (label_v2 + 1) // 2
        labels = label_v2[np.newaxis]

        image = tf.compat.v1.io.read_file(img_dir)
        image = tf.compat.v1.image.decode_png(image, 3)
        image_v2 = image[np.newaxis]
        image_v2 = tf.compat.v1.image.resize(image_v2, [args["load_size"], args["load_size"]])
        image_v2 = tl.center_crop(image_v2, size=args["crop_size"])
        fin_image = tf.compat.v1.clip_by_value(image_v2, 0, 255) / 127.5 - 1

    def sample_graph():

        Genc, Gdec, _ = src.src_AttGAN.module.get_model(args["model"], n_atts, weight_decay=args["weight_decay"])

        xa = tf.compat.v1.placeholder(tf.float32, shape=[None, args["crop_size"], args["crop_size"], 3])
        b_ = tf.compat.v1.placeholder(tf.float32, shape=[None, n_atts])

        x = Gdec(Genc(xa, training=False), b_, training=False)

        save_dir = './output/%s/samples_testing_%s' % (experiment_name, '{:g}'.format(test_int))
        py.mkdir(save_dir)

        def run():
            cnt = 0

            for _ in tqdm.trange(1):
                xa_ipt, a_ipt = sess.run([fin_image, labels])

                b_ipt_list = [a_ipt]
                for i in range(n_atts):
                    tmp = np.array(a_ipt, copy=True)
                    tmp[:, i] = 1 - tmp[:, i]

                    b_ipt_list.append(tmp)

                x_opt_list = [xa_ipt]
                for i, b_ipt in enumerate(b_ipt_list):
                    b__ipt = (b_ipt * 2 - 1).astype(np.float32)
                    if i > 0:
                        b__ipt[:,i - 1] = b__ipt[:,i - 1] * test_int
                    x_opt = sess.run(x, feed_dict={xa: xa_ipt, b_: b__ipt})
                    x_opt_list.append(x_opt)

                sample = np.transpose(x_opt_list, (1, 2, 0, 3, 4))
                sample = np.reshape(sample, (sample.shape[0], -1, sample.shape[2] * sample.shape[3], sample.shape[4]))

                small_attr_dict = {
                    0:"Input",
                    1:"Reconstruction",
                    2:"Bald" ,
                    3:"Bangs" ,
                    4:"Black_Hair",
                    5:"Blond_Hair",
                    6:"Brown_Hair",
                    7:"Bushy_Eyebrows" ,
                    8:"Eyeglasses",
                    9:"Male",
                    10:"Mouth_Slightly_Open",
                    11:"Mustache",
                    12:"No_Beard",
                    13:"Pale_Skin",
                    14:"Young"
                }

                for s in sample:
                    for i in range(0,15):
                        image_v2 = s[:,128*i:128*(i+1),:]
                        im.imwrite(image_v2, f'{save_dir}/{small_attr_dict[i]}.jpg')
                        im.imwrite(s, f'{save_dir}/all_images.jpg')

        return run

    """sample = sample_graph()
    
    # checkpoint
    if not os.path.exists(py.join(output_dir, 'generator.pb')):
        checkpoint = tl.Checkpoint(
            {v.name: v for v in tf.compat.v1.global_variables()},
            py.join(output_dir, 'checkpoints'),
            max_to_keep=1
        )
        checkpoint.restore().run_restore_ops()
    
    sample()"""
    sess.close()