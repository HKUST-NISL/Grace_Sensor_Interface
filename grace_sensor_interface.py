#general
import yaml
import rospy
import os
import re
import threading
from signal import signal
from signal import SIGINT
import logging
import sys
from datetime import datetime
import time
from inspect import getsourcefile
from os.path import abspath

#ros
import dynamic_reconfigure.client
import sensor_msgs.msg
import std_msgs.msg
import hr_msgs.msg
import grace_attn_msgs.msg
import grace_attn_msgs.srv
import hr_msgs.msg
import hr_msgs.cfg
import hr_msgs.srv
import std_msgs

#Misc
file_path = os.path.dirname(os.path.realpath(getsourcefile(lambda:0)))
sys.path.append(os.path.join(file_path, '..'))
from CommonConfigs.grace_cfg_loader import *
from CommonConfigs.logging import setupLogger

#Respond to exit signal
def handle_sigint(signalnum, frame):
    # terminate
    print('Main interrupted! Exiting.')
    sys.exit()



class SensorInterface:

    __latest_word = ''#Not used

    __start_faking = False
    __latest_interim = None
    __latest_interim_time_stamp = 0

    def __init__(self, config_data):
        #miscellaneous
        signal(SIGINT, handle_sigint)

        #Config
        self.__config_data = config_data

        #Sensor interface uses its own logger at its own dir
        self.__logger = setupLogger(
                    logging.DEBUG, 
                    logging.DEBUG, 
                    self.__class__.__name__,
                    os.path.join(file_path,"./logs/log_") + datetime.now().strftime(self.__config_data['Custom']['Logging']['time_format']))

        self.__nh = rospy.init_node(self.__config_data['Sensors']['Ros']['node_name'])

        #Ros io
        self.__asr_words_sub = rospy.Subscriber(
                                self.__config_data['HR']['ASRVAD']['asr_words_topic'], 
                                hr_msgs.msg.ChatMessage, 
                                self.__asrWordsCallback, 
                                queue_size=self.__config_data['Custom']['Ros']['queue_size'])
        self.__asr_interim_sub = rospy.Subscriber(
                                self.__config_data['HR']['ASRVAD']['asr_interim_speech_topic'], 
                                hr_msgs.msg.ChatMessage, 
                                self.__asrInterimCallback, 
                                queue_size=self.__config_data['Custom']['Ros']['queue_size'])
        self.__asr_fake_sentence_pub = rospy.Publisher(
                                self.__config_data['HR']['ASRVAD']['asr_fake_sentence_topic'], 
                                hr_msgs.msg.ChatMessage, 
                                queue_size=self.__config_data['Custom']['Ros']['queue_size'])


        #Camera configs
        self.__cam_ang_sub = rospy.Subscriber(
                                self.__config_data['Custom']['Sensors']['topic_set_cam_angle'], 
                                std_msgs.msg.Float32, 
                                self.__cameraAngCallback, 
                                queue_size=self.__config_data['Custom']['Ros']['queue_size'])
        self.__cam_cfg_client = dynamic_reconfigure.client.Client(
            self.__config_data['HR']['Cam']['camera_cfg_server'],
            timeout= self.__config_data['Custom']['Ros']['dynam_config_timeout'])
        self.setCameraAngle(self.__config_data['Sensors']['Vision']['default_grace_chest_cam_angle'])


        #ASR Configs
        self.__fake_sentence_rate = self.__config_data['Sensors']['ASR']['asr_fake_sentence_check_rate']
        self.__fake_sentence_window = self.__config_data['Sensors']['ASR']['asr_fake_sentence_window']
        self.__prime_lang = self.__config_data['Sensors']['ASR']['primary_language_code']
        self.__second_lang = self.__config_data['Sensors']['ASR']['secondary_language_code']
        self.__asr_model = self.__config_data['Sensors']['ASR']['asr_model']
        self.__continuous_asr = self.__config_data['Sensors']['ASR']['asr_continuous']
        self.__asr_reconfig_client = dynamic_reconfigure.client.Client(
                        self.__config_data['HR']['ASRVAD']['asr_reconfig'],
                        timeout= self.__config_data['Custom']['Ros']['dynam_config_timeout'])
                         
        #VAD configs
        self.__vad_dynamic_config_client = dynamic_reconfigure.client.Client(
                            self.__config_data['HR']['ASRVAD']['vad_config'], 
                            timeout= self.__config_data['Custom']['Ros']['dynam_config_timeout'])


        #Initialize asr
        self.__asrInit()
        self.__vadInit()




    '''
        VISION-CAM-ROS Helpers
    '''
    def __cameraAngCallback(self,msg):
        self.setCameraAngle(msg.data)





    '''
    #   ASR-ROS-Helpers
    '''
    def __asrInit(self):
        #We turn off first
        params = { 'enable': False} 
        self.__asr_reconfig_client.update_configuration(params)
        #Then restart
        params = { 
            'enable': True, 
            'language': self.__prime_lang, 
            'alternative_language_codes': self.__second_lang, 
            'model': self.__asr_model, 
            'continuous': self.__continuous_asr,
            'asr_activity_mon': self.__config_data['Sensors']['ASR']['asr_activity_monitor']
            } 
        self.__asr_reconfig_client.update_configuration(params)
        #Start the fake sentence generator
        self.__fake_sentence_thread = threading.Thread(target = self.__fakeSentenceThread, daemon=False)
        self.__fake_sentence_thread.start()

    def __asrWordsCallback(self, msg):
        self.__latest_word = msg.utterance
        self.__logger.debug('Latest WORD: (%s).' % self.__latest_word)

    def __asrInterimCallback(self, msg):
        #Receive the latest asr string
        self.__latest_interim = msg
        self.__latest_interim_for_bardging_in = self.__latest_interim.utterance
        self.__logger.debug('Latest INTERIM %s' %{self.__latest_interim.utterance})

        #Upon receiving a new interim sentence, we update the timestamp and start faking sentences
        self.__start_faking = True
        self.__latest_interim_time_stamp = rospy.get_time()
        
    def __fakeSentenceThread(self):
        rate = rospy.Rate(self.__fake_sentence_rate)

        while True:
            rate.sleep()

            if( self.__start_faking ):#If we have started to fake a sentence
                #Check the timestamp of the latest interim speech
                if( rospy.get_time() - self.__latest_interim_time_stamp >= self.__fake_sentence_window ):
                    #Publish a fake sentence  
                    self.__asr_fake_sentence_pub.publish(self.__latest_interim)

                    #Log
                    self.__logger.info('Publishing FAKE asr sentence %s' % {self.__latest_interim.utterance})

                    #Reset the fields
                    self.__start_faking = False
                    self.__latest_interim = None

    def __vadInit(self):
        self.__vad_dynamic_config_client.update_configuration(
                                            {
                                                "enabled":self.__config_data['Sensors']['VAD']['enabled'], 
                                                "continuous": self.__config_data['Sensors']['VAD']['continuous'],
                                                "language": self.__config_data['Sensors']['VAD']['vad_lang'],
                                                "vad_confidence": self.__config_data['Sensors']['VAD']['vad_confidence'],
                                                "vad_sensitivity": self.__config_data['Sensors']['VAD']['vad_sensitivity']
                                            }
                                        )


    #Interface
    def setCameraAngle(self, angle):
		#tilt chest cam to a given angle
        try:
            self.__logger.info("Configuring camera angle to %f." % angle)

            self.__cam_cfg_client.update_configuration({"motor_angle":angle})
        except Exception as e:
            self.__logger_error(e)

    def mainLoop(self):
        rate = rospy.Rate(1)

        while True:
            rate.sleep()


if __name__ == '__main__':
    grace_config = loadGraceConfigs()
    sensor_interface = SensorInterface(grace_config)
    sensor_interface.mainLoop()


