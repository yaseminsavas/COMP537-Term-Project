# COMP537 PROJECT
IRA: Image Retouching with Audio

IRA presents a voice-based, personalized user interface to enable the retouching process as the smoothest.

3 steps in the project:

1. Initial Suggestions based on the differences between the original image and the AttGAN resulting images
2. Intent Classification (Now Keyword - Action Matching)
3. AttGAN to convert the image based on user input

Tasks Performed:

- Interface Implementation
- RASA NLU Training
- Multi-feature AttGAN training
- Model Development & Improvement
- Suggestions pop-up
- User Evaluation

To run the project:

1. Use requirements.txt for the needed packages

2. Please select an image from CelebA-HQ dataset & place it in your local (images outside this dataset works to a certain level too, but use images from the dataset for good results) Some example images from CelebA-HQ: (Drive link: https://drive.google.com/drive/folders/1Vljq4FhbaI1Gorr149xX-mj0Xc-kKahL?usp=sharing )
3. Please unzip the Checkpoints&Summaries.zip file (Drive link: https://drive.google.com/drive/folders/1Vljq4FhbaI1Gorr149xX-mj0Xc-kKahL?usp=sharing ) and place the checkpoints and summaries files under the output/AttGAN_128_CelebA-HQ **separately**.
4. Run the main.py file

- Note1: For all images in CelebA-HQ, refer this link: https://github.com/switchablenorms/CelebAMask-HQ
- Note2: If you want to download all of the dataset, I followed the steps in this repository for data & used the pre-given model weights: https://github.com/LynnHo/AttGAN-Tensorflow

References:

[1] Zhenliang He, Wangmeng Zuo, Meina Kan, Shiguang Shan, and Xilin Chen. 2019. Attgan: Facial attribute editing by only changing what you want. IEEE Transactions on Image Processing 28, 11 (2019), 5464–5478. DOI:http://dx.doi.org/10.1109/tip.2019.2916751

[2] Cheng-Han Lee et al. “MaskGAN: Towards Diverse and Interactive Facial Image Manipula-
tion”. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2020.
