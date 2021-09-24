from mycroft import MycroftSkill, intent_file_handler
import cozmo
import time
import random

class Cozmo(MycroftSkill):
    def initialize(self):
        self.bus.on('speak', self.handle_speech)

    def handle_speech(self, message):
        global cSay
        tts_string = message.data['utterance']
        self.log.info(tts_string)
        print(f'tts string {tts_string}')
    #    if len(tts_string) > 255: # looks like cozmo cant take 256. todo add this

        cSay = str(tts_string)
        time.sleep(.1)
        cozmo.run_program(self.cozmo_program)
        cozmo.run_program(self.cozmo_animate)

    def cozmo_program(self, robot: cozmo.robot.Robot):
        global cSay
        robot.say_text(cSay).wait_for_completed()


    def cozmo_animate(self, robot: cozmo.robot.Robot):
        # grab a list of animation triggers
        all_animation_triggers = robot.anim_triggers

        # randomly shuffle the animations
        random.shuffle(all_animation_triggers)


        # select the animations from the shuffled list
        triggers = 1
        chosen_triggers = [trigger for trigger in robot.anim_triggers if 'WinGame' in trigger.name]
        # some of the animations do not disconnect. Todo find the lust of broken animations
        #chosen_triggers = all_animation_triggers[:triggers] #again more fun but some hang
        print('Playing {} random animations:'.format(triggers))

        # play the number of triggers animations one after the other, waiting for each to complete
        for trigger in chosen_triggers:
            print('Playing {}'.format(trigger.name))
            robot.play_anim_trigger(trigger).wait_for_completed()


        # grab animation triggers that have 'WinGame' in their name
        chosen_triggers = [trigger for trigger in robot.anim_triggers if 'WinGame' in trigger.name]
        def __init__(self):
            MycroftSkill.__init__(self)

        @intent_file_handler('cozmo.intent')
        def handle_cozmo(self, message):
            self.speak_dialog('cozmo')


def create_skill():
    return Cozmo()

