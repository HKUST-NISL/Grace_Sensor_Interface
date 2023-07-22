import collections, queue
import numpy as np
import pyaudio
import webrtcvad
from halo import Halo
import torch
import torchaudio
import logging
from datetime import datetime
import os
from inspect import getsourcefile
from os.path import abspath
import sys
import time
from signal import signal
from signal import SIGINT
import threading

#ros
import rospy
import dynamic_reconfigure.client
import sensor_msgs.msg
import std_msgs.msg
import hr_msgs.msg
import hr_msgs.msg
import std_msgs
import audio_common_msgs.msg

#Misc
file_path = os.path.dirname(os.path.realpath(getsourcefile(lambda:0)))
sys.path.append(os.path.join(file_path, '..'))
from CommonConfigs.grace_cfg_loader import *
from CommonConfigs.logging import setupLogger



def handle_sigint(signalnum, frame):
    # terminate
    print('Main interrupted! Exiting.')
    sys.exit()


def Int2Float(sound):
    _sound = np.copy(sound)  #
    abs_max = np.abs(_sound).max()
    _sound = _sound.astype('float32')
    if abs_max > 0:
        _sound *= 1/abs_max
    audio_float32 = torch.from_numpy(_sound.squeeze())
    return audio_float32




class GraceVAD:

    def __init__(self, config_data, logger) -> None:
        #Config and logging
        self.__config_data = config_data
        self.__logger = logger.getChild(self.__class__.__name__)



        #VAD model
        torchaudio.set_audio_backend(self.__config_data['Sensors']['SileroVAD']['audio_backend'])
        self.__model, self.__utils = torch.hub.load(
                            repo_or_dir = self.__config_data['Sensors']['SileroVAD']['model'],
                            model=self.__config_data['Sensors']['SileroVAD']['model_name'],
                            force_reload = self.__config_data['Sensors']['SileroVAD']['force_reload'])
        (self.__get_speech_ts,_,_,_,_) = self.__utils

        #Ros IO
        self.__vad_pub = rospy.Publisher(
            self.__config_data['Custom']['Sensors']['topic_silero_vad_name'],
            std_msgs.msg.String,
            queue_size= self.__config_data['Custom']['Ros']['queue_size']
        )
        self.__raw_audio_sub = rospy.Subscriber(
            self.__config_data['Custom']['Sensors']['topic_silero_vad_raw_audio'],
            audio_common_msgs.msg.AudioData,
            self.rawAudioCallback,
            queue_size= 1
        )
        self.__vad_conf_sub = rospy.Subscriber(
            self.__config_data['Custom']['Sensors']['topic_silero_vad_conf_thresh_name'],
            std_msgs.msg.Float32,
            self.vadConfThreshCallback,
            queue_size= 1
        )



        #Initial conf threshold
        self.__vad_conf_thresh = self.__config_data['Sensors']['SileroVAD']['conf_threshold']


    def vadConfThreshCallback(self,msg):
        self.__vad_conf_thresh = msg.data
        self.__logger.info('VAD thresh updated to %f.' % (self.__vad_conf_thresh) )



    def rawAudioCallback(self, msg):

        #Pre-process chunk
        newsound= np.frombuffer(msg.data,np.int16)
        audio_float32=Int2Float(newsound)

        #Run vad over the chunk
        time_stamps = self.__get_speech_ts(
                                audio_float32, 
                                self.__model,
                                #VAD configs
                                threshold = self.__vad_conf_thresh,
                                sampling_rate  = self.__config_data['Sensors']['SileroVAD']['sampling_rate'],
                                min_speech_duration_ms = self.__config_data['Sensors']['SileroVAD']['min_speech_dur_ms'],
                                max_speech_duration_s = self.__config_data['Sensors']['SileroVAD']['max_speech_dur_s'],
                                min_silence_duration_ms = self.__config_data['Sensors']['SileroVAD']['min_silence_dur_ms'],
                                window_size_samples = self.__config_data['Sensors']['SileroVAD']['internal_window_size_samples'],
                                speech_pad_ms = self.__config_data['Sensors']['SileroVAD']['speech_padding_ms'],
                                )
        
        #Check if there is a speech in this chunk
        if(len(time_stamps)>0):
            self.__vad_pub.publish(self.__config_data['Sensors']['SileroVAD']['speech_string'])
            self.__logger.info("Silero VAD: speech")
        else:
            self.__vad_pub.publish(self.__config_data['Sensors']['SileroVAD']['non_speech_string'])
            self.__logger.info("Silero VAD: non-speech")




if __name__ == '__main__':
    signal(SIGINT, handle_sigint)
    