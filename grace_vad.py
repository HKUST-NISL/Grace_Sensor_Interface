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


#Misc
file_path = os.path.dirname(os.path.realpath(getsourcefile(lambda:0)))
sys.path.append(os.path.join(file_path, '..'))
from CommonConfigs.grace_cfg_loader import *
from CommonConfigs.logging import setupLogger

#Config and logging
config_data = loadGraceConfigs()

logger = setupLogger(
        logging.DEBUG, 
        logging.INFO, 
        __name__,
        os.path.join(file_path,"./logs/log_") 
        + config_data['Sensors']['SileroVAD']['node_name']
        + datetime.now().strftime(config_data['Custom']['Logging']['time_format']))




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

'''
#For this part refers to silero-vad
'''

class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS):
        def proxy_callback(in_data, frame_count, time_info, status):
            #pylint: disable=unused-argument
            callback(in_data)
            return (None, pyaudio.paContinue)
        if callback is None: callback = lambda in_data: self.buffer_queue.put(in_data)
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
        self.pa = pyaudio.PyAudio()

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }

        self.chunk = None
        # if not default device
        if self.device:
            kwargs['input_device_index'] = self.device

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)

class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, aggressiveness=3, device=None, input_rate=None):
        super().__init__(device=device, input_rate=input_rate)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            raise Exception("Resampling required")

    def vadFrameStream(self, frames=None):
        '''
            Maintains a sliding window over the audio stream and 
            invoke silero vad to process the audio chunk at a certain "rough" frequency
        '''
        if frames is None: frames = self.frame_generator()

        for frame in frames:
            if len(frame) < 640:
                return            
            yield frame

#Initial conf threshold
vad_conf_thresh = config_data['Sensors']['SileroVAD']['conf_threshold']
def vadConfThreshCallback(msg):
    global vad_conf_thresh 
    vad_conf_thresh = msg.data
    logger.info('VAD thresh updated to %f.' % (vad_conf_thresh) )


#Audio frame storage
ring_buffer = None
def vadProcThread():

    #audio frame buffer
    global ring_buffer

    #VAD model
    torchaudio.set_audio_backend(config_data['Sensors']['SileroVAD']['audio_backend'])
    model, utils = torch.hub.load(
                        repo_or_dir = config_data['Sensors']['SileroVAD']['model'],
                        model=config_data['Sensors']['SileroVAD']['model_name'],
                        force_reload = config_data['Sensors']['SileroVAD']['force_reload'])
    (get_speech_ts,_,_,_,_) = utils

    #Ros IO
    vad_pub = rospy.Publisher(
        config_data['Custom']['Sensors']['topic_silero_vad_name'],
        std_msgs.msg.String,
        queue_size= config_data['Custom']['Ros']['queue_size']
    )

    vad_conf_sub = rospy.Subscriber(
        config_data['Custom']['Sensors']['topic_silero_vad_conf_thresh_name'],
        std_msgs.msg.Float32,
        vadConfThreshCallback,
        queue_size= 1
    )

    #Loop and process
    rate = rospy.Rate(hz=config_data['Sensors']['SileroVAD']['yield_freq_hz'])
    while True:
        rate.sleep()

        #Compose binary data array
        wav_data = bytearray()
        for f in ring_buffer:
            wav_data.extend(f)

        #Pre-process chunk
        newsound= np.frombuffer(wav_data,np.int16)
        audio_float32=Int2Float(newsound)

        #Run vad over the chunk
        time_stamps =get_speech_ts(
                                audio_float32, 
                                model,
                                #VAD configs
                                threshold = vad_conf_thresh,
                                sampling_rate  = config_data['Sensors']['SileroVAD']['sampling_rate'],
                                min_speech_duration_ms = config_data['Sensors']['SileroVAD']['min_speech_dur_ms'],
                                max_speech_duration_s = config_data['Sensors']['SileroVAD']['max_speech_dur_s'],
                                min_silence_duration_ms = config_data['Sensors']['SileroVAD']['min_silence_dur_ms'],
                                window_size_samples = config_data['Sensors']['SileroVAD']['internal_window_size_samples'],
                                speech_pad_ms = config_data['Sensors']['SileroVAD']['speech_padding_ms'],
                                )
        
        #Check if there is a speech in this chunk
        if(len(time_stamps)>0):
            vad_pub.publish(config_data['Sensors']['SileroVAD']['speech_string'])
            logger.info("Silero VAD: speech")
        else:
            vad_pub.publish(config_data['Sensors']['SileroVAD']['non_speech_string'])
            logger.info("Silero VAD: non-speech")


def main():
    # Ros routine
    nh = rospy.init_node(config_data['Sensors']['SileroVAD']['node_name'])

    # Initialize the audio object
    vad_audio = VADAudio(
                    aggressiveness=config_data['Sensors']['SileroVAD']['webRTC_aggressiveness'],
                    device=None,#Use default device
                    input_rate=config_data['Sensors']['SileroVAD']['sampling_rate'])
    
    #Audio frame stream
    frames = vad_audio.vadFrameStream()

    # A sliding window for storing audio frames
    num_frames_window = config_data['Sensors']['SileroVAD']['window_size_ms'] // vad_audio.frame_duration_ms
    global ring_buffer
    ring_buffer = collections.deque(maxlen=num_frames_window)

    # Start streaming and the processing thread
    proc_thread = threading.Thread(target=vadProcThread)
    logger.info("Begin listening (ctrl-C to exit)...")
    proc_thread.start()
    for frame in frames:
        ring_buffer.append(frame)



if __name__ == '__main__':
    signal(SIGINT, handle_sigint)
    main()