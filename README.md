# Cinder
$ mycroft cinder transformers distilgpt2 cozmo robot


#Install mycroft
sudo apt install git
git clone https://github.com/MycroftAI/mycroft-core.git
bash dev_setup.sh
    
# Install transformers to run the distilgpt2model
sudo apt install python3-pip
git clone https://github.com/huggingface/transformers.git
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
mycroft-msk create
mycroft-pip install transformers
mycroft-pip install torch
  
# For Cozmo Skill 
mycroft-pip install 'cozmo[camera]'
mycroft-msk create

# Changing the wake word
mycroft-config edit user
mycroft-config reload
  
#Running Mycroft with Cozmo adb has to be on with the Cozmo app enabled in SDK mode.
adb server start
./start-mycroft.sh debug   
./stop-mycroft.sh all

#######################################################################################################
