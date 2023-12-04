import pyaudio
import speech_recognition as sr
import audioop
import math
import tempfile
import os
import wave
import subprocess
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from gtts import gTTS
import pygame

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Initialize the recognizer
recognizer = sr.Recognizer()

# Set up audio stream parameters
input_device_index = None
sample_rate = 22050  # Reduced sample rate
chunk_size = 8192  # Increased chunk size (in bytes)
threshold_db = 45  # Adjusted threshold

# Create a temporary audio file
temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
temp_audio_file_name = temp_audio_file.name
temp_audio_file.close()

# Open an input audio stream
input_stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=sample_rate,
    input=True,
    frames_per_buffer=chunk_size,
    input_device_index=input_device_index
)

print("Listening...")

# Initialize variables for voice activity detection
audio_data = bytearray()
speech_started = False

commands = {
    'home': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'map'}, pose: {position: {x: 3.3491172790527344, y: -0.004497826099395752, z: 0.002140045166015625}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
    'room1': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'map'}, pose: {position: {x: 7.17515230178833, y: 0.668110728263855, z: 0.0033092498779296875}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
    'Room 1': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'map'}, pose: {position: {x: 7.17515230178833, y: 0.668110728263855, z: 0.0033092498779296875}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
    'Room One': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'map'}, pose: {position: {x: 7.17515230178833, y: 0.668110728263855, z: 0.0033092498779296875}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
    'room one': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'map'}, pose: {position: {x: 7.17515230178833, y: 0.668110728263855, z: 0.0033092498779296875}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
    'room 2': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'map'}, pose: {position: {x: 9.780485153198242, y: -4.3226823806762695, z: 0.0046672821044921875}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
    'room two': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'map'}, pose: {position: {x: 9.780485153198242, y: -4.3226823806762695, z: 0.0046672821044921875}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
    'room 3': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'map'}, pose: {position: {x: 3.4306554794311523, y: -7.267694473266602, z: 0.0068721771240234375}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
    'room three': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'map'}, pose: {position: {x: 3.4306554794311523, y: -7.267694473266602, z: 0.0068721771240234375}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
    'room 4': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'map'}, pose: {position: {x: -1.6239104270935059, y: 0.7523196935653687, z: 0.00223541259765625}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
    'room four': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'map'}, pose: {position: {x: -1.6239104270935059, y: 0.7523196935653687, z: 0.00223541259765625}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
    'room 5': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'map'}, pose: {position: {x: 1.6131210327148438, y: 2.954784393310547, z: 0.00484466552734375}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
    'room five': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'map'}, pose: {position: {x: 1.6131210327148438, y: 2.954784393310547, z: 0.00484466552734375}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'"
}
  
"""
    'five nine five': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped" \ 
'{{header:" \ 
  "stamp:" \
    "sec: 0" \
    "nanosec: 0" \
  "frame_id: 'map'" \
"pose:" \
  "position:"\
    "x: 0.0" \
    "y: 0.0"\
    "z: 0.0"\
  "orientation:"\
    "x: 0.0"\
    "y: 0.0"\
    "w: 1.0}}'"  }
"""

# Load MiniLM model and tokenizer
#model_name = "bert-base-uncased"
#model = AutoModelForSequenceClassification.from_pretrained(model_name)
#tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the voice outputs dictionary
voice_outputs = {
    'home': "I will take you home",
    'room 1': "I will take you to room 1",
    'room one': "I will take you to room 1",
    'room One': "I will take you to room 1",
    'room 1': "I will take you to room 1",
    'room 2': "I will take you to room 2",
    'room two': "I will take you to room 2",
    'room 3': "I will take you to room 3",
    'room three': "I will take you to room 3",
    'room 4': "I will take you to room 4",
    'room four': "I will take you to room 4",
    'room 5': "I will take you to room 5",
    'room five': "I will take you to room 5"
}


try:
    while True:
        audio_chunk = input_stream.read(chunk_size, exception_on_overflow=False)
        audio_data.extend(audio_chunk)

        rms = audioop.rms(audio_chunk, 2)
        decibel = 20 * math.log10(rms) if rms > 0 else 0

        if decibel > threshold_db:
            if not speech_started:
                print("Speech Started")
                speech_started = True
        else:
            if speech_started:
                print("Speech Ended")

                with open(temp_audio_file_name, "wb") as f:
                    wav_header = wave.open(temp_audio_file_name, 'wb')
                    wav_header.setnchannels(1)
                    wav_header.setsampwidth(2)
                    wav_header.setframerate(sample_rate)
                    wav_header.writeframes(audio_data)
                    wav_header.close()

                with sr.AudioFile(temp_audio_file_name) as source:
                    try:
                        transcription = recognizer.record(source)
                        recognized_text = recognizer.recognize_google(transcription)
                        if recognized_text:
                            print("Transcription: " + recognized_text)

                            for cmd in commands.keys():
                                if cmd in recognized_text:
                                    print('CHECKING COMMANDS')
                                    ros_command = commands[cmd]
                                    print(commands[cmd])
                                    

                                    # Use gTTS to convert the voice output to speech and play it
                                    if recognized_text in voice_outputs:
                                        print("Found voice")
                                        response = voice_outputs[cmd]
                                        tts = gTTS(text=response, lang='en', slow=False)
                                        tts.save("temp_audio.mp3")
                                        os.system("mpg321 temp_audio.mp3")
                                        print("Audio file Made")
                                    
                                    # Play the audio file using pygame
                                    pygame.mixer.init()
                                    pygame.mixer.music.load("temp_audio.mp3")
                                    pygame.mixer.music.play()
                                    
                                    # Send the ROS message using subprocess
                                    subprocess.check_output(ros_command, shell=True, stderr=subprocess.STDOUT)

                    except sr.UnknownValueError:
                        print("No speech detected")
                    except sr.RequestError as e:
                        print(f"Could not request results; {e}")

                speech_started = False
                audio_data = bytearray()

except KeyboardInterrupt:
    pass

input_stream.stop_stream()
input_stream.close()
os.remove(temp_audio_file_name)
audio.terminate()
