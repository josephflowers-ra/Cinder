# My process for installing mycroft, transformers, and cozmo sdk on Ubuntu 20.04.3 LTS.
# Then creating two skills to use mycroft to capture speach and send it as input to the 
# model and send the output from the model to cozmo.
# This is a work in progress and I will update it as I work on it.
# Cozmo skill __init__.py in the cozmo-skill folder uses wingame animations by default
# as they do not hang up. The random is more interesting but a few animations are non functional.
# There is a list of the non functional ones around somewhere and I will add that to a delete
# list to remove them.
#
# I will find a place to upload the model as its too big for GitHub and share the link here.
#
# I also included my testing chat script for interfacing with the model.
#

#Install mycroft
sudo apt install git
git clone https://github.com/MycroftAI/mycroft-core.git
bash dev_setup.sh
    
# Install transformers to run the distilgpt2model
sudo apt install python3-pip
git clone https://github.com/huggingface/transformers.git
#next command needs to be run in the transformers directory
pip install -q ./transformers
pip install datasets
pip install torch
     
# Install cozmo
sudo apt install python-tk
sudo apt-get install python3-pil.imagetk
pip3 install --user 'cozmo[camera]'
git clone https://github.com/anki/cozmo-python-sdk.git

# Install adb server to connect to phone app
sudo apt install adb
adb start-server

# For Cinder skill
mycroft-pip install transformers
mycroft-pip install torch
mycroft-msk create

# For Cozmo Skill 
mycroft-pip install 'cozmo[camera]'
mycroft-msk create

# Changing the wake word. The mycroft.conf file has what I used here.
mycroft-config edit user
mycroft-config reload
  
#Running Mycroft with Cozmo adb has to be on with the Cozmo app enabled in SDK mode.
adb server start
./start-mycroft.sh debug   
#stopping mycroft
./stop-mycroft.sh all

#######################################################################################################
