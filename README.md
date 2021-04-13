# Topical Lectures, Controls 2021

Andreas Freise, Bas Swinkels 13.04.2021

This repository provides the material for the hands-on session on control system.

All the work in this hands-on project will be done with Python inside Jupyter notebooks. We provide you with several notebooks which contain task descriptions and suggestions. You will complete the task by working inside the notebooks.

## Installing Jupyter and Python
We expect you to have installed Python and Jupyter as described in: [SoftwareInstall.pdf](https://raw.githubusercontent.com/freise/controls_2021/main/SoftwareInstall.pdf)

Download all files from this repository, store them in the same folder and open the notebooks listed below with Jupyter.

If you cannot install Python and Jupyter on your own computer you can still follow parts of the course by using [Google Colab](#colab), however this is not recommended and we cannot provide help with the Google Colab system.

## The course material

The material is divided into 6 notebooks that you should work through in sequence:

 * **student0_example_notebook.ipynb**: A simple example, to step you through the basics of using a Jupyter notebook. If you are already familiar with Jupyter you should skip this part. 
 * **student1_keyboard.ipynb**: A quick interactive demonstration of the drone model code.
 * **student2_system_identification.ipynb**: In this notebook you will perform experiments to measure specific parameters of your virtual drone.
 * **student3_basic_control.ipynb**: Now we design and test simple feedback control systems for the drone.
 * **student4a_racing.ipynb**: To find out if the controls are fit for purpose: race your drone through a track and compare your time with others!
 * **student4b_racing_noninteractive.ipynb**: This is a non-interactive version of the race (which is a bit more difficult).

## <a name="colab"></a> Running the notebooks in Google Colab

If you cannot install Python and Jupyter on your own computer, you can run some of the provided notebooks online, using the Google Colab project. You will need a Google account to save your work to Google Drive.

You cannot run the following notebooks because Colab does not provide the QT interface for interactive plots:
 - student1_keyboard.ipynb
 - student4a_racing.ipynb

The following notebooks will work:
- student0_example_notebook.ipynb
- student2_system_identification.ipynb
- student3_basic_control.ipynb
- student4_advanced_control.ipynb
- student4b_racing_noninteractive.ipynb

However, you must add a new code cell at the top of each notebook and add (and execute) this command:

```!wget https://raw.githubusercontent.com/freise/controls_2021/main/module.py```

Follow this link to open one of the notebooks in Google Colab:

https://colab.research.google.com/github/freise/controls_2021/
