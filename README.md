# IRA_COMP537_PROJECT
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
3. Please unzip the Checkpoints&Summaries.zip file (Drive link: https://drive.google.com/drive/folders/1Vljq4FhbaI1Gorr149xX-mj0Xc-kKahL?usp=sharing ) and place the checkpoints and summaries files under the output/AttGAN_128_CelebA-HQ separately.
4. Run the main.py file

- Note1: For all images in CelebA-HQ, refer this link: https://github.com/switchablenorms/CelebAMask-HQ
- Note2: If you want to download all of the dataset, I followed the steps in this repository for the dataset: https://github.com/LynnHo/AttGAN-Tensorflow