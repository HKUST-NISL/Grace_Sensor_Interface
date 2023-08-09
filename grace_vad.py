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
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model
import scipy.io.wavfile as wf
import wave
import io

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



        #Silero VAD model
        if(self.__config_data['Sensors']['SileroVAD']['enabled']):
            torchaudio.set_audio_backend(self.__config_data['Sensors']['SileroVAD']['audio_backend'])
            self.__model, self.__utils = torch.hub.load(
                                repo_or_dir = self.__config_data['Sensors']['SileroVAD']['model'],
                                model=self.__config_data['Sensors']['SileroVAD']['model_name'],
                                force_reload = self.__config_data['Sensors']['SileroVAD']['force_reload'])
            (self.__get_speech_ts,_,_,_,_) = self.__utils
            #Initial conf threshold
            self.__silero_vad_conf_thresh = self.__config_data['Sensors']['VAD']['conf_threshold']

            self.__silero_vad_conf_sub = rospy.Subscriber(
                self.__config_data['Custom']['Sensors']['topic_vad_conf_thresh'],
                std_msgs.msg.Float32,
                self.vadConfThreshCallback,
                queue_size= 1
            )



        #Pyannote vad
        if(self.__config_data['Sensors']['PyannoteVAD']['enabled']):
            if(self.__config_data['Sensors']['PyannoteVAD']['pipeline_opt'] == 0):
                self.__pyannote_pipeline = Pipeline.from_pretrained(
                                os.path.join(getConfigPath(),self.__config_data['Sensors']['PyannoteVAD']['pipeline_vad']),
                                use_auth_token=self.__config_data['Sensors']['PyannoteVAD']['hf_token'])
            
            elif(self.__config_data['Sensors']['PyannoteVAD']['pipeline_opt'] == 1):
                # self.__pyannote_model = Model.from_pretrained(
                #                 'pyannote/segmentation', 
                #                 use_auth_token=self.__config_data['Sensors']['PyannoteVAD']['hf_token'])
                # self.__pyannote_pipeline = VoiceActivityDetection(segmentation=self.__pyannote_model)
                # HYPER_PARAMETERS = {
                #     # onset/offset activation thresholds
                #     "onset": 0.85, "offset": 0.5,
                #     # remove speech regions shorter than that many seconds.
                #     "min_duration_on": 0,
                #     # fill non-speech regions shorter than that many seconds.
                #     "min_duration_off": 0.0
                # }
                # self.__pyannote_pipeline.instantiate(HYPER_PARAMETERS)
                pass
            
            elif(self.__config_data['Sensors']['PyannoteVAD']['pipeline_opt'] == 2):
                #TBD
                pass

            else:
                pass

        #Ros IO
        self.__vad_pub = rospy.Publisher(
            self.__config_data['Custom']['Sensors']['topic_vad_results'],
            std_msgs.msg.String,
            queue_size= self.__config_data['Custom']['Ros']['queue_size']
        )
        self.__raw_audio_sub = rospy.Subscriber(
            self.__config_data['Custom']['Sensors']['topic_vad_raw_audio'],
            audio_common_msgs.msg.AudioData,
            self.rawAudioCallback,
            queue_size= 1
        )



    def vadConfThreshCallback(self,msg):
        if(self.__config_data['Sensors']['SileroVAD']['enabled']):
            self.__silero_vad_conf_thresh = msg.data

            self.__logger.info('VAD thresh updated to %f.' % (self.__silero_vad_conf_thresh) )



    def rawAudioCallback(self, msg):

        #Pre-process chunk
        newsound= np.frombuffer(msg.data,np.int16)

        '''
        Run vad over the chunk
        '''
        vad_flag = False
        if(self.__config_data['Sensors']['PyannoteVAD']['enabled']):
            # # Pyannote vad
            wf.write(
                    self.__config_data['Sensors']['PyannoteVAD']['tmp_file_name'], 
                    self.__config_data['Sensors']['VAD']['sampling_rate'], 
                    newsound)
            output = self.__pyannote_pipeline(self.__config_data['Sensors']['PyannoteVAD']['tmp_file_name'])
            
            if(self.__config_data['Sensors']['PyannoteVAD']['pipeline_opt'] == 0):
                # This is the routine for vad pipeline
                segments = output.get_timeline().support()
                vad_flag = len(segments)>0

            elif(self.__config_data['Sensors']['PyannoteVAD']['pipeline_opt'] == 1):
                # # This is the routine for segmentation pipeline
                # self.__logger.info(output._labels)         
                pass

            elif(self.__config_data['Sensors']['PyannoteVAD']['pipeline_opt'] == 2):
                # This is the routine for diarization pipeline

                #TBD
                pass

                # vad_flag = len(output._labels) > 1
                # self.__logger.debug('Processing vad results.')
                # for label in output._labels:
                #     # speaker speaks between turn.start and turn.end
                #     self.__logger.debug('Speaker %s.' % (label) )
                
            else:
                self.__logger.error('Unexpected pipeline choice.')




        if(self.__config_data['Sensors']['SileroVAD']['enabled']):
            # # Silero vad
            audio_float32=Int2Float(newsound)
            time_stamps = self.__get_speech_ts(
                                    audio_float32, 
                                    self.__model,
                                    #VAD configs
                                    threshold = self.__silero_vad_conf_thresh,
                                    sampling_rate  = self.__config_data['Sensors']['VAD']['sampling_rate'],
                                    min_speech_duration_ms = self.__config_data['Sensors']['SileroVAD']['min_speech_dur_ms'],
                                    max_speech_duration_s = self.__config_data['Sensors']['SileroVAD']['max_speech_dur_s'],
                                    min_silence_duration_ms = self.__config_data['Sensors']['SileroVAD']['min_silence_dur_ms'],
                                    window_size_samples = self.__config_data['Sensors']['SileroVAD']['internal_window_size_samples'],
                                    speech_pad_ms = self.__config_data['Sensors']['SileroVAD']['speech_padding_ms'],
                                    )
            vad_flag = len(time_stamps)>0
        
        #Check if there is a speech in this chunk
        if(vad_flag):
            self.__vad_pub.publish(self.__config_data['Sensors']['VAD']['speech_string'])
            self.__logger.info("VAD: %s" % (self.__config_data['Sensors']['VAD']['speech_string']) )
        else:
            self.__vad_pub.publish(self.__config_data['Sensors']['VAD']['non_speech_string'])
            self.__logger.info("VAD: %s" % (self.__config_data['Sensors']['VAD']['non_speech_string']) )




if __name__ == '__main__':
    signal(SIGINT, handle_sigint)
    