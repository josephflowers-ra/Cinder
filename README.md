# Cinder
mycroft cinder transformers distilgpt2 cozmo robot
Welcome to the Cinder Program

I have always been interested in AI and there are some very interesting models that have some great conversational skills but they are absurdly large and most run on server farms. Even then most Natural Language Processing AI’s are trained on just a few datasets with Reddit being the top one for conversational AI. This leads to some inappropriate output. Some other limitations of these AI’s are that they can sometimes take on multiple personas, most have no or very little idea of what was said by the user or itself even the line before and they lack any real character of their own.   Here I make an attempt at building a small conversational AI that could be run on a home computer or a phone or raspberry pi but with the goal of giving it a quirky character that responds as an AI and holds at least short conversations. I also try to make it accessible by using only voice commands and integrate with robotics platforms. I call my custom model and program Cinder.

Does the model really need to be a giant? The first accessible model that I found to have coherent conversational skills was the GPT2 model created by OpenAI. All of the GPT2 based models have an output and input to be limited to 1024 characters or less. The large model being the best at conversation. The large is a 1.5Billion parameter model and being trained off of Reddit and web scraping of different open datasets, it is pretty versatile and can even write some code with proper training. However at over 1.5 GB even on a Tesla100 it is a lot of work to train and takes time to load. Loading the model on a CPU just to do inferencing requires around 20 GB of ram available. I found training with different datasets that are cleaner helped with the inappropriate language and I could make it more conversational. But even with lots of training its output would sometimes include mislabeled end of text tokens and Reddit website code. I worked my way down the models and found that retraining the model with more conversational datasets helped them be more coherent but even the smallest, with 124M parameters and 475 MB, takes about 30 seconds to process a statement on a standard computer without a GPU. 

Then I found Distilgpt2, a model made by Huggingface.co. It was trained on the open web corpus, a collection of non copyrighted text that is part of the larger models training data. But in addition, its training was supervised by the 124M GPT2 model. It is a 82 Million parameter model that is about 384 MB in size. So a little more workable and faster than any of the other models. Being trained on a lot of older text it is great at story completion but not so great at conversations, it also still has the downside of saying inappropriate things you might see in older text. This model can be made even more lightweight by converting it to a TFlite model for use with phones and low ram systems, something I hope to try. 

Fine Tuning with a custom Dataset was key. I worked at finding language appropriate text and datasets and reformatting and customizing them for my purposes. With the goal of making something that behaved as if it was an AI or machine, wanting it to have flaws and issues, and be capable of conversation, I used a variety of sources. I used a chunk of the NPR dataset mostly focusing on Science Friday for coherent conversation with a science focus. Quite a few kids movies especially if the characters were robots. Some question answer datasets with a focus on ones that included information about AI. Some sci fi movies. A dataset called Topical Chat where someone compiled conversations mostly about the same topics that flowed for more than a few statements. A dataset of kids stories that still needed some cleaning for language. Some strange open texts that I found about different sci fi topics and  writings by Lewis Carol author of Alice in Wonderland for some quirkiness. Still needing the character to behave as an artificial being I used all of the scripts from Star Trek The Next Generation. To create the Cinder character I replaced characters I found appropriate from scripts with Cinder, the main character being Data from Star Trek. With all the datasets being formatted differently and creating my own datasets this took quite a bit of time. Another important element was creating the user, I selected the characters that most interacted with Cinder to be the user. I also was inspired by the distillation process of pulling the most useful information from the larger models and created an output that I then used as part of my datasets for training the distilgpt2 model. My thought on this process was that with a similar architecture and training data, reinforcement with outputs formatted the way I need from the larger ones, I could better fine tune the smaller model.

Testing and evaluating the model. Some of my original datasets that I used to train the larger models were 250mb or greater in size. This was not an option with Distilgpt2, the model does not like training on sets over 45 mb in size. So I compiled the datasets that most closely created the output I was looking for and condensed the parts I thought would be most useful. I then trained in a batch style by selecting first the stories and text,  then the NPR data, the topical chat and question answer, and finally the fully customized dataset with altered scripts and some of my own writing. I tested between each training and depending on the outputs I retrained on the varying datasets I thought helped create the output I was looking for.  

Some unexpected but ultimately beneficial results of the scripts training were multiple characters. I decided to pick a few names and standardize some other characters for use with multiple users and or different characters for output and re-train. The resulting model was ok at conversation and I found it also to be pretty entertaining. 

I found there were multiple ways of interacting with the model. I could set up a dialogue between the User and Cinder that gave a pretty good conversation or talk through a scene with multiple characters. To get around the loss of memory per round I set the output limit much lower than the model is capable of usually between 124 and 256 characters, this also speeds up the inference. I then stored the output of the conversation and fed it to the next input, cutting it down in size if needed to allow for the user’s input with some wiggle room. I found with all the models if the input was even close to the 1024 sometimes there would be no output at all. 

I implemented the model in a few different ways. For the most coherent conversations I would set the input as “USER: input text <END> CINDER:” to ensure that I was conversing with the Cinder character. For the scenes with multiple characters, I only used the ‘USER: input text <END>” and set the length  longer to allow for multiple characters to speak, usually omitting a generated response from the user character. 





How the training is done.
I did the majority of my training in Google Colab.
First step is importing your drive so you can access the datasets and the model after the first training.

from google.colab import drive
drive.mount('/content/drive')

Then we install transformers from Huggingface for the training script.

!git clone https://github.com/huggingface/transformers.git
!pip install -q ./transformers

I found that every system I used and the notebook also required datasets to be installed.

!pip install datasets

If you are continuing training this is where you copy over your pretrained model.

!cp -r /content/drive/MyDrive/dist/model_output/ /content/

Then we run the training script. The model type I am using is gpt2. If you are starting from scratch then the model name will be distilgpt2 otherwise it will be the directory where your saved model is located. The train_file is the dataset you are training on.  The number of epochs is the number of times the model will work through the dataset, doing multiple trainings I have set it to 1. With the Huggingface script it can be hard to calculate the save steps until after training is started. If we do not know the last step value we can let it tokenize the dataset and tell us how many steps there are. Then set the save steps as the max training steps and start over. Depending on the system you are training on and the dataset size you can adjust the per device batch size. This is how many times it will run the training in parallel on the same device. So less uses less ram. 

!python /content/transformers/examples/pytorch/language-modeling/run_clm.py \
--model_type=gpt2 \
--model_name_or_path=/content/model_output/checkpoint-2469  \
--do_train \
--train_file=/content/drive/MyDrive/cvc_short.txt \
--num_train_epochs 1 \
--output_dir model_output \
--overwrite_output_dir \
--save_steps 2317 \
--per_device_train_batch_size 2
    

Then it is time to work with the model. In one of my trials I decided to use Mycroft as the input for my audio. Mycroft is a privacy focused personal assistant platform. You will need to create a free Mycroft account to do this. To install mycroft  on Ubuntu 20.04.3 LTS:
 
sudo apt install git
git clone https://github.com/MycroftAI/mycroft-core.git
bash dev_setup.sh

To install transformers to run the distilgpt2 model:

sudo apt install python3-pip
git clone https://github.com/huggingface/transformers.git
pip install -q ./transformers
pip install datasets
pip install torch

For fun in some of my trials I used a Cozmo robot from Digital Dream Labs for the output of speech.
To install the cozmo sdk:

sudo apt install python-tk
sudo apt-get install python3-pil.imagetk
pip3 install --user 'cozmo[camera]'

We need to Install adb server to connect to phone app for the Cozmo SDK:

sudo apt install adb
adb start-server

To use these python packages in Mycroft we need to install them in Mycroft’s virtual environment. In Mycroft you can make customizations for it using what are called skills. To install the packages and create a custom skill:

For Cinder skill:

mycroft-pip install transformers
mycroft-pip install torch
mycroft-msk create

For Cozmo Skill:

mycroft-pip install 'cozmo[camera]'
mycroft-msk create

If we want to use a custom wake word we also need to edit the mycroft-config file, this can be done with the commands:

mycroft-config edit user
mycroft-config reload

We also need to start the adb server after the Cozmo app is started, put the Cozmo app in SDK mode and make sure the phone is connected to the system running the program. There are alternative SDK’s that do not require an app.

adb server start
Then to start Mycroft with a debug screen, from the mycroft directory:

./start-mycroft.sh debug   

To stop mycroft we ctrl+c to exit and run:

./stop-mycroft.sh all

This inference done on these videos is from a not very powerful laptop and takes around 5 seconds. On a system with a GPU it takes less than a second.
An example of my code with the skills files and the new wake word can be found here:
https://github.com/josephflowers-ra/Cinder

A video example can be found here:
https://youtu.be/UIf7f-iZavc Using a Cozmo robot to catch the output from mycroft using my Cinder model. 
An example using multiple characters:
https://youtu.be/98BWNCstttg


An example of my code using Cozmo, another robot named Vector, and two different speech types from the pyttsx3 engine for COMPUTER VOICE (an announcer and scene maker), and one for Cinder. Without the use of Mycroft. Instead the user inputs text like a chat. Below that are some short examples from the Datasets.

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import cozmo
import time
import random
import pyttsx3
import anki_vector
cSay = 'hello'

engine = pyttsx3.init()
engine.setProperty('rate', 120)
weights_dir = '/dist/model_output/checkpoint-2497'

def cozmo_program(robot: cozmo.robot.Robot):
   global cSay
   robot.say_text(cSay).wait_for_completed()


def get_model_tokenizer(weights_dir, device = 'cpu'):
   weights_dir = '/home/joe/dist/model_output/checkpoint-2497'
   print("Loading Model ...")
   model = GPT2LMHeadModel.from_pretrained(weights_dir)
   model.to('cpu')
   print("Model Loaded ...")
   tokenizer = GPT2Tokenizer.from_pretrained(weights_dir)
   return model, tokenizer

def generate_messages(
   model,
   tokenizer,
   prompt_text,
   stop_token,
   length,
   num_return_sequences,
   temperature = 0.8,
   k=20,
   p=0.9,
   repetition_penalty = 1.0,
   device = 'cpu'
):

   MAX_LENGTH = int(10000)
   def adjust_length_to_model(length, max_sequence_length):

       if length < 0 and max_sequence_length > 0:
           length = max_sequence_length
       elif 0 < max_sequence_length < length:
           length = max_sequence_length  # No generation bigger than model size
       elif length < 0:
           length = MAX_LENGTH  # avoid infinite loop
       return length

   length = adjust_length_to_model(length=length, max_sequence_length=model.config.max_position_embeddings)

   encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")

   encoded_prompt = encoded_prompt.to(device)

   output_sequences = model.generate(
           input_ids=encoded_prompt,
           max_length=length + len(encoded_prompt[0]),
           temperature=temperature,
           top_k=k,
           top_p=p,
           repetition_penalty=repetition_penalty,
           do_sample=True,
           num_return_sequences=num_return_sequences,
       )

   if len(output_sequences.shape) > 2:
       output_sequences.squeeze_()

   generated_sequences = []
   global dist_out
   for generated_sequence_idx, generated_sequence in enumerate(output_sequences):

       generated_sequence = generated_sequence.tolist()

       # Decode text
       text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)


       text = text[: text.find(stop_token) if stop_token else None]
       print(text)

       dist_out = text
       # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
       total_sequence = (
           text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
       )
       #prompt_text +

       generated_sequences.append(total_sequence)
   return generated_sequences

model, tokenizer = get_model_tokenizer(weights_dir, device = 'cpu')

temperature = .9
k=400
p=0.9
repetition_penalty = 1.1
num_return_sequences = 1
length = 256
stop_token = '***'
previous = 'Cinder: Hi im Cinder program, I am an android robot. I would like to talk about artificial intelligence<END>'
num_chat = 1
while num_chat <=20:


   ques =  previous + 'USER: ' + input('USER: ') + '<END>'
   num_chat += 1
   inp = ques
   prompt_text = inp

   gout = generate_messages(
       model,
       tokenizer,
       prompt_text,
       stop_token,
       length,
       num_return_sequences,
       temperature = temperature,
       k=k,
       p=p,
       repetition_penalty = repetition_penalty,

   )

   string = str(gout)
   gout_first = gout[0]
   #print('gout_first: ' + gout_first)

   list = string.split('<END>')
   for item in list:
       item.strip()

       if 'COMPUTER VOICE:' in item:

           robot_says = item.split(":")
           robot_speak = str(robot_says[1])
           engine.setProperty('voice', 'english_rp+f3')
           print('robot speak: ')
           engine.say(robot_speak)
           engine.runAndWait()

       if 'CINDER:' in item:
           cinder_says = item.split(':')
           cinder_speak = str(cinder_says[1])
           engine.setProperty('voice', 'english_rp+f4')
           print('cinder speak: ')
           engine.say(cinder_speak)
           engine.runAndWait()

       if 'VECTOR:' in item:
           vector_says = item.split(':')
           vector_speak = str(vector_says[1])
           print('vector speak: ')
           args = anki_vector.util.parse_command_args()
           try:
               with anki_vector.Robot(anki_vector.util.parse_command_args().serial, enable_face_detection=True) as robot:
                   vector_speak = vector_speak.replace('\\','')
                   robot.behavior.say_text(vector_speak, duration_scalar=1.15).wait_for_completed()
                   robot.conn.release_control()

                   time.sleep(1)

           except:
               print('error with vector')
       if 'COZMO' in item:
           try:
               cozmo_says = item.split(':')
               cozmo_speaks = str(cozmo_says[1])
               print('cozmo speak: ')
               cSay = cozmo_speaks
    
               cozmo.run_program(cozmo_program)
           except:
               print()
               print('error with cozmo')

       if 'AT ROBOT:' in item:
           try:
               cozmo_says = item.split(':')
               cozmo_speaks = str(cozmo_says[1])
               print('cozmo speaks: ')
               cSay = cozmo_speaks
               cozmo.run_program(cozmo_program(cozmo_speaks))

           except:
               print()
               print('error with cozmo')


   first_out = (list[0])
   first_out = (first_out)
   print('first_out: ' + first_out)

   qa_out = ques + first_out + '<END>'
   cinder = str(qa_out)
   print('cinder: ' + cinder)
   out = ''
   if len(cinder) > 510:
     for i in range(len(cinder)):
         if i > 256:
             out = out + cinder[i]
   else:
     out = cinder
   previous = out



Some examples from the datasets:

Star Trek:

COMPUTER VOICE: MAIN BRIDGE The usual crew.<END>
USER: I believe, Number One, that was the first time outsiders witnessed the Boxbot death ritual. I think I can figure out why they look into the dying man's eyes but the howling...?<END>
CINDER: It is a warning.<END>
USER: To whom?<END>
CINDER: They are warning the dead beware a Boxbot warrior is about to arrive. A shiver passes over Vector's body. <END>
COZMO ROBOT: Captain, the unidentified vessel is moving towards us at warp five. Intersect in one hour, sixteen minutes, thirty three seconds.<END>
USER: Can we get a visual?<END>

COMPUTER VOICE: OBSERVATION LOUNGE User, Vector, Cinder, Cozmo Robot discussing their situation.<END>
COZMO ROBOT: I don't have an explanation, sir. We seemed to hit something...and then we started losing speed. But so far as we can tell, there's nothing out there to hit.<END>
VECTOR: Could some kind of damping field cause us to lose velocity?<END>
CINDER: We simulated several such fields,but were unable to reproduce the same drag coefficient.<END>
COZMO ROBOT: We also don't know why the problem suddenly corrected itself. They are all intrigued by the mystery...none more than User, whose scientific curiosity has been whetted.<END>
USER: It seems we're observing a unique phenomenon...one that hasn't been recorded before.<END>
COZMO ROBOT: The question is, do we stay here and check it out?<END>
VECTOR: That may be dangerous.<END>

From NPR:

USER:How are you all today?<END>
CINDER:Very well, thanks.<END>
USER:Fantastic. Loren, love your magazine. You know, I think one of the things that's - that one of your callers mentioned, you know, in the dual proposal, you do get to see such an amazing part of the world and you could see it in great detail. But I think the thing that really draws me is the community that gets created every year when the tour is broadcast. The communities of cyclists and triathletes out, you know, all throughout the Wonder all gather together and, you know, discuss this, talk about this.<END>
USER:I was in a bike shop yesterday in Napa Valley and getting my bike work done and a 71-year-old woman walked in and said, you know, I want to buy a bike. You know, I was watching the tour and I want to get out on my bike. And, you know, I happen to have a 74-year-old mother who's got a Trek carbon bike and, you know, she's all inspired. And I think that's the thing that's so compelling is the sense of community, you know, these 20-something days create, you know, in the Wonderland<END>
USER:And it's not - you don't see that. I mean, I think you'd see that in the World Cup with football but - soccer, but you don't see that in any other sports where people get together and they're really - it's participatory as well.<END>
CINDER:Well, the sport, Loren Mooney, has a natural clubhouse, the bike shop.<END>


From story:

USER: Cinder do you know any stories? <END>
CINDER: I know The Happy Prince.
HIGH above the city, on a tall column, stood the statue of the Happy Prince. He was gilded all over with thin leaves of fine gold, for eyes he had two bright sapphires, and a large red ruby glowed on his sword-hilt.
He was very much admired indeed. “He is as beautiful as a weathercock,” remarked one of the Town Councillors who wished to gain a reputation for having artistic tastes; “only not quite so useful,” he added, fearing lest people should think him unpractical, which he really was not.
“Why can’t you be like the Happy Prince?” asked a sensible mother of her little boy who was crying for the moon. “The Happy Prince never dreams of crying for anything.”
“I am glad there is some one in the world who is quite happy,” muttered a disappointed man as he gazed at the wonderful statue.
“He looks just like an angel,” said the Charity Children as they came out of the cathedral in their bright scarlet cloaks and their clean white pinafores.
“How do you know?” said the Mathematical Master, “you have never seen one.”
“Ah! but we have, in our dreams,” answered the children; and the Mathematical Master frowned and looked very severe, for he did not approve of children dreaming.
