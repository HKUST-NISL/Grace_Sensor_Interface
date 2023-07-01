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





#Load configs
def loadConfig(path):
    #Load configs
    with open(path, "r") as config_file:
        config_data = yaml.load(config_file, Loader=yaml.FullLoader)
        # print("Config file loaded")
    return config_data

#Create Logger
def setupLogger(file_log_level, terminal_log_level, logger_name, log_file_name):
    log_formatter = logging.Formatter('%(asctime)s %(msecs)03d %(name)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s', 
                                  datefmt='%d/%m/%Y %H:%M:%S')

    f = open(log_file_name, "a")
    f.close()
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(file_log_level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(terminal_log_level)

    logger = logging.getLogger(logger_name)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel( min(file_log_level,terminal_log_level) )#set to lowest

    return logger

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

    def __init__(self):
        #miscellaneous
        signal(SIGINT, handle_sigint)
        self.__logger = setupLogger(
                    logging.DEBUG, 
                    logging.DEBUG, 
                    self.__class__.__name__,
                    "./logs/log_" + datetime.now().strftime("%a_%d_%b_%Y_%I_%M_%S_%p"))
        path = "./config/config.yaml"
        self.__config_data = loadConfig(path)
        self.__nh = rospy.init_node(self.__config_data['Ros']['node_name'])

        #Ros io
        self.__asr_words_sub = rospy.Subscriber(self.__config_data['Ros']['asr_words_topic'], hr_msgs.msg.ChatMessage, self.__asrWordsCallback, queue_size=self.__config_data['Ros']['queue_size'])
        self.__asr_interim_sub = rospy.Subscriber(self.__config_data['Ros']['asr_interim_speech_topic'], hr_msgs.msg.ChatMessage, self.__asrInterimCallback, queue_size=self.__config_data['Ros']['queue_size'])
        self.__asr_fake_sentence_pub = rospy.Publisher(self.__config_data['Ros']['asr_fake_sentence_topic'], hr_msgs.msg.ChatMessage, queue_size=self.__config_data['Ros']['queue_size'])
        self.__asr_reconfig_client = dynamic_reconfigure.client.Client(self.__config_data['Ros']['asr_reconfig']) 

        #Camera configs
        self.__cam_cfg_client = dynamic_reconfigure.client.Client(self.__config_data['Ros']['camera_cfg_server'])
        self.setCameraAngle(self.__config_data['Vision']['default_grace_chest_cam_angle'])

        #ASR Configs
        self.__fake_sentence_rate = self.__config_data['ASR']['asr_fake_sentence_check_rate']
        self.__fake_sentence_window = self.__config_data['ASR']['asr_fake_sentence_window']
        self.__prime_lang = self.__config_data['ASR']['primary_language_code']
        self.__second_lang = self.__config_data['ASR']['secondary_language_code']
        self.__asr_model = self.__config_data['ASR']['asr_model']
        self.__continuous_asr = self.__config_data['ASR']['asr_continuous']

        #Initialize asr
        self.__asrInit()




    '''
        VISION-CAM-ROS Helpers
    '''






    '''
    #   ASR-ROS-Helpers
    '''
    def __asrInit(self):
        #We turn off first
        params = { 'enable': False} 
        self.__asr_reconfig_client.update_configuration(params)
        #Then restart
        params = { 'enable': True, 'language': self.__prime_lang, 'alternative_language_codes': self.__second_lang, 'model': self.__asr_model, 'continuous': self.__continuous_asr} 
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
    sensor_interface = SensorInterface()
    sensor_interface.mainLoop()


