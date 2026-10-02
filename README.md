# Information Theory of Written Text and Animal Vocalization Annotation Practical

## Overview
This assignment consists of two parts that require Python experience. Complete **Part 1 first**, then proceed to Part 2.

## Setup Options

### Option 1: Local Installation (Recommended)
Clone the repository and set up your Python environment using either:
- **venv**: `python -m venv env` then activate with `source env/bin/activate` (Mac/Linux) or `env\Scripts\activate` (Windows)

```bash
pip install -r requirements.txt
```

- **miniconda**: Create a new conda environment

```bash
conda create --name nccr_it --file requirements.txt
conda activate nccr_it
```


### Option 2: Google Colab
If you prefer not to install locally, use these Colab links:

Part 1: [<a href="https://colab.research.google.com/drive/12ad5FFXDzTa5ZRRHtc50JWicWoUg2dGF?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>]


## Assignment Parts
### Part 1: Entropy of Written Text

Open part1_entropy_written_text.ipynb
Uses text_analysis.py helper functions
Complete instructions are in the notebook -- there are some questions to deepen your insights into information theory applied to language

### Part 2: Introduction to VoCallBase and CallMark
Navigate to the website - https://vocallbase2.evolvinglanguage.ch/?tab=welcome
<img width="1920" height="1040" alt="image" src="https://github.com/user-attachments/assets/96b65ea6-dc62-4397-b1d2-251ace27da70" />

Set up an account by going to the Account Drop-down menu and hitting sign-up - 
<img width="549" height="182" alt="image" src="https://github.com/user-attachments/assets/004276cb-726d-4b0c-8070-fe6293a395fc" />

Fill out the credentials, selecting the Waterslager canary dataset to start - 
<img width="486" height="532" alt="image" src="https://github.com/user-attachments/assets/967bb0d7-f8ee-43d8-876a-b74935a4c040" />

Save the authentication token somewhere and press continue (sign-in) -
<img width="478" height="299" alt="image" src="https://github.com/user-attachments/assets/dae95275-ba63-49ce-8a00-05bfc79e27a7" />

Navigate to the user account dropdown menu and select dashboard - 
<img width="756" height="255" alt="image" src="https://github.com/user-attachments/assets/1fcc66d2-ca42-48e2-8bb8-41fa0cd2b5b5" />

Select the Waterslager Canary Song dataset - 
<img width="1088" height="441" alt="image" src="https://github.com/user-attachments/assets/8238036c-dd51-40ea-9621-8a12bc086f71" />

For a given audio file, select "View" to see annotations done by a domain expert or "Annotate" to make your own annotations - 
<img width="1354" height="421" alt="image" src="https://github.com/user-attachments/assets/95558596-0c04-40c4-93fe-28ef57c672ed" />

In the View tab, select "Open" - 
<img width="1078" height="450" alt="image" src="https://github.com/user-attachments/assets/350f3d90-28b3-4d21-8bc7-554568906682" />

This will bring you to the CallMark interface - 
<img width="1422" height="718" alt="image" src="https://github.com/user-attachments/assets/dae4c710-776a-4e78-b144-0e3fd1151222" />

Press the play button (above Spectrogram Parameter) to listen to audio, select the right and left arrows to navigate time (top right) in the clip, and you can save the annotations to a csv file (top left) -
<img width="386" height="291" alt="image" src="https://github.com/user-attachments/assets/e0ac3aba-00aa-491b-bd19-cd630ffb3408" />

Return to the Browse Files menu and select "Annotate" to make your own annotations - 
<img width="1354" height="421" alt="image" src="https://github.com/user-attachments/assets/95558596-0c04-40c4-93fe-28ef57c672ed" />

In the new CallMark interface, select the "Add new species" button to create a new class of labels - 
<img width="455" height="79" alt="image" src="https://github.com/user-attachments/assets/98ea02a3-f09e-4922-8386-9b7c41f00ede" />

Make it align with what is seen from the domain expert annotations from the View menu (species - "canaries" and individual can be identified from the filename) - 
<img width="723" height="76" alt="image" src="https://github.com/user-attachments/assets/198385e9-0215-4be6-84f1-c6d3d685ac3d" />

Select the individual and vocalizations classes you created - 
<img width="726" height="74" alt="image" src="https://github.com/user-attachments/assets/87c58a11-c5a7-4d9f-92b6-caf2f3424e35" />

You will see the new annotation classes appear below the spectrogram - 
<img width="395" height="87" alt="image" src="https://github.com/user-attachments/assets/f88184ff-c525-4f38-8c0b-ad0252b35ae7" />

You may now use that class of annotations to select the onsets and offsets by left clicking where you see and hear those classes of annotations occur on the spectrogram - 
<img width="1337" height="602" alt="image" src="https://github.com/user-attachments/assets/7de11860-5829-488a-8e54-0a1fdaa9189e" />

To make fine-adjustments to an annotation click and hold on the onset-offset vertical lines tweak them. To delete an annotation, right click on the annotation below the spectrogram viewer - 
<img width="177" height="541" alt="image" src="https://github.com/user-attachments/assets/af7596d0-3122-46be-be1c-2b94cf2d78bd" />

Once you are done annotating, press the "Annotated Area" button to indicate where you have annotated, increasing the spectrogram view slider makes this easier if you have annotated an entire clip -
<img width="1913" height="617" alt="image" src="https://github.com/user-attachments/assets/2a82d511-1f5b-42be-8035-5fe9ba4a49de" />

If you wish to compare your annotations to those of a domain expert, go ahead and download your annotations -
<img width="386" height="291" alt="image" src="https://github.com/user-attachments/assets/e0ac3aba-00aa-491b-bd19-cd630ffb3408" />






