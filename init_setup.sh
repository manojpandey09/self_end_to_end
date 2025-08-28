#!/bin/bash

echo [$(date)]: "START"  # Script start ho raha hai

# Python 3.8 environment banane ke liye command
echo [$(date)]: "Creating env with Python 3.8 version"
conda create --prefix ./env python=3.8 -y  # #envBanana #Python38 #condaUse

# Environment activate karna jaruri hai taaki usi environment mein packages install ho
echo [$(date)]: "Activating the environment"
source /c/Users/ASUS/anaconda/etc/profile.d/conda.sh  # #condaSetup #EnvironmentActivate
conda activate ./env  # #activateEnv

# Ab dev requirements install karenge jo project chalane ke liye chahiye
echo [$(date)]: "Installing the dev requirements"
pip install -r requirements.txt  # #requirementsInstall #devSetup

echo [$(date)]: "END"  # Script end ho gaya
