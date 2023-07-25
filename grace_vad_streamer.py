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



#Config and logging
config_data = loadGraceConfigs()

logger = setupLogger(
        logging.DEBUG, 
        logging.INFO, 
        __name__,
        os.path.join(file_path,"./logs/log_") 
        + config_data['Sensors']['VAD']['streamer_node_name']
        + datetime.now().strftime(config_data['Custom']['Logging']['time_format']))

#Sliding window of audio frame
ring_buffer = None



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
        '''
        if frames is None: frames = self.frame_generator()

        for frame in frames:
            if len(frame) < 640:
                return            
            yield frame


buffer_lock = threading.Lock()
def streamingThread():
    global buffer_lock
    raw_audio_pub = rospy.Publisher(
        config_data['Custom']['Sensors']['topic_vad_raw_audio'],
        audio_common_msgs.msg.AudioData,
        queue_size= 1
    )


    rate = rospy.Rate(config_data['Sensors']['VAD']['yield_freq_hz'])

    while True:
        rate.sleep()

        wav_data = bytearray()
        buffer_lock.acquire()
        for f in ring_buffer:
            wav_data.extend(f)
        buffer_lock.release()
        msg = audio_common_msgs.msg.AudioData()
        msg.data = list(wav_data)
        raw_audio_pub.publish(msg)
        logger.info('Streamining wav audio data.')




def main():

    # Ros routine
    nh = rospy.init_node(config_data['Sensors']['VAD']['streamer_node_name'])


    # Initialize the audio object
    vad_audio = VADAudio(
                    aggressiveness=config_data['Sensors']['SileroVAD']['webRTC_aggressiveness'],
                    device=None,#Use default device
                    input_rate=config_data['Sensors']['VAD']['sampling_rate'])
    
    #Audio frame stream
    frames = vad_audio.vadFrameStream()

    # A sliding window for storing audio frames
    global ring_buffer
    num_frames_window = config_data['Sensors']['VAD']['window_size_ms'] // vad_audio.frame_duration_ms
    ring_buffer = collections.deque(maxlen=num_frames_window)

    # Start streaming 
    streaming_thread = threading.Thread(target=streamingThread)
    logger.info("Begin listening (ctrl-C to exit)...")
    streaming_thread.start()
    
    for frame in frames:
        buffer_lock.acquire()
        ring_buffer.append(frame)
        buffer_lock.release()

if __name__ == '__main__':
    signal(SIGINT, handle_sigint)
    main()