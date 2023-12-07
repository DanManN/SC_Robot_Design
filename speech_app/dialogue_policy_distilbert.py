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
import torch
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
import warnings
warnings.simplefilter("ignore")

navigation_phrases = ['take', 'walk', 'guid']
verbal_directions_phrases = ['point', 'tell', 'find', 'giv', 'how']

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Initialize the recognizer
recognizer = sr.Recognizer()

# Set up audio stream parameters
input_device_index = None
sample_rate = 22050  # Reduced sample rate
chunk_size = 8192  # Increased chunk size (in bytes)
threshold_db = 50  # Adjusted threshold

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
'room 1': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'map'}, pose: {position: {x: 7.17515230178833 y: 0.668110728263855 z: 0.0033092498779296875}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
'room 2': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'map'}, pose: {position: {x: 9.780485153198242 y: -4.3226823806762695 z: 0.0046672821044921875}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
'room 3': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'map'}, pose: {position: {x: 3.4306554794311523 y: -7.267694473266602 z: 0.0068721771240234375}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
'room 4': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'map'}, pose: {position: {x: -1.6239104270935059 y: 0.7523196935653687 z: 0.00223541259765625}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
'room 5': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'map'}, pose: {position: {x: 1.6131210327148438 y: 2.954784393310547 z: 0.00484466552734375}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
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
distilbert_model_name = "distilbert-base-cased-distilled-squad"
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(distilbert_model_name)
distilbert_model = DistilBertForQuestionAnswering.from_pretrained(distilbert_model_name)


def distilbert_question_answering(question, context):
    # Tokenize input
    inputs = distilbert_tokenizer(question, context, return_tensors="pt", truncation=True)

    # Get the model's output
    outputs = distilbert_model(**inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Get the answer span
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    # Convert token indices to actual tokens
    tokens = distilbert_tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
    answer_tokens = tokens[start_index:end_index+1]

    # Convert tokens back to text
    answer = distilbert_tokenizer.decode(distilbert_tokenizer.convert_tokens_to_ids(answer_tokens))

    return answer.strip()




# Define the voice outputs dictionary
nav_outputs = {
    '1': "I will take you to room 1",
    '2': "I will take you to room 2",
    '3': "I will take you to room 3",
    '4': "I will take you to room 4",
    '5': "I will take you to room 5",
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

                            intent = intent_classifier(recognized_text)

                            if intent == 0: #walk navigation
                                for cmd in commands.keys():
                                    if cmd in recognized_text:
                                        print('CHECKING COMMANDS')
                                        ros_command = commands[cmd]
                                        print(ros_command)
                                        speech_output_gen(recognized_text)

                                        # Send the ROS message using subprocess
                                        subprocess.check_output(ros_command, shell=True, stderr=subprocess.STDOUT)
                                        

                            elif intent == 1: ##verbal directions
                                question = recognized_text
                                context = """
                                You are a helpful robot designed to assist people in navigating our large meeting room. The room has been structured with several distinct locations to facilitate various activities. Allow me to provide you with detailed directions:

                                - Location 1: Positioned by the window, it offers a scenic view and is close to the emergency exit on the eastern side.
                                - Location 2: Situated by the door closest to the lab equipment, making it convenient for accessing laboratory resources.
                                - Location 3: Adjacent to the big surfing structure, an artistic installation that serves as a focal point in the room.
                                - Location 4: Near the secondary entrance on the western side, providing an alternative access point.
                                - Location 5: The middle of the room, marked by a central meeting area with comfortable seating arrangements.

                                Example Question: How can I get from the middle of the room to location 1?
                                Example Answer:
                                To get from the middle of the room (Location 5) to Location 1, follow these directions:
                                1. Head east, passing by the central meeting area.
                                2. Continue towards the window on the eastern side.
                                3. Once you reach the window, you will have arrived at Location 1.  # Example: This is an example of the answer format

                                Feel free to ask for directions to any specific location, and I'll be happy to assist you!
                                """
                                answer = distilbert_question_answering(question, context)
                                print('Predicted answer:', answer)


                            else:   #misc
                                speech_output_gen(recognized_text)  


                            # Play the audio file using pygame
                            pygame.mixer.init()
                            pygame.mixer.music.load("temp_audio.mp3")
                            pygame.mixer.music.play()
    

                    except sr.UnknownValueError:
                        print("No speech detected")
                    except sr.RequestError as e:
                        print(f"Could not request results; {e}")

                speech_started = False
                audio_data = bytearray()

except KeyboardInterrupt:
    pass

def intent_classifier(utt:str):
    ##user wants to be walked
    if any(nav_phrase in utt for nav_phrase in navigation_phrases):
        print('******* IN NAVIGATION CONDITION')
        return 0
    ##user wants verbal directions
    elif any(v_phrase in utt for v_phrase in verbal_directions_phrases):
        print('******* IN verbal CONDITION')
        return 1

    ##user wants to chat
    else:
        print('******* IN CHAT CONDITION')
        return 2

def speech_output_gen(utt:str):
    # Use gTTS to convert the voice output to speech and play it
    if utt in nav_outputs:
        print("Found voice")
        response = nav_outputs[cmd]
        tts = gTTS(text=response, lang='en', slow=False)
        tts.save("temp_audio.mp3")
        os.system("mpg321 temp_audio.mp3")
        print("Audio file Made")

    else:
        print('IN MISC SPeech output')
        response = "Please be sure to select a valid room, and I can walk you there or give you directions."
        tts = gTTS(text=response, lang='en', slow=False)
        tts.save("temp_audio.mp3")
        os.system("mpg321 temp_audio.mp3")
        print("Audio file Made")


input_stream.stop_stream()
input_stream.close()
os.remove(temp_audio_file_name)
audio.terminate()
