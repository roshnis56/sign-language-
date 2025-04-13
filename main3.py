import cv2
import mediapipe as mp
import numpy as np
import pickle
import google.generativeai as genai

from PIL import Image, ImageTk
import tkinter as tk
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from gtts import gTTS
import os
from mutagen.mp3 import MP3
from googletrans import Translator
import pygame
import time
from warnings import filterwarnings

filterwarnings('ignore')
translator = Translator()
# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Global variables
capturing = False
captured_signs = []
complete_sentence = ""  # Initialize complete_sentence variable
recognized_letter = ""
recognized_word = ""

# Load ASL model
with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)

# Configure and initialize the Generative AI model
genai.configure(api_key='AIzaSyCaz-8RhG6tW1SuLq5L_QINLTm9neK9dX0')
model = genai.GenerativeModel('gemini-1.5-pro')
chat = model.start_chat(history=[])


# Function to capture and append sign
def capture_sign():
    global capturing
    capturing = True

def stop_capture(nn, lan):
    global capturing, captured_signs, complete_sentence
    capturing = False
    if captured_signs:
        complete_word = ''.join(captured_signs)
        print("Recognized Words:", complete_word)

        # Removing stopwords and generating POS tags
        tokens = word_tokenize(complete_word.lower())
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

        # Get POS tags
        pos_tags = pos_tag(filtered_tokens)

        # Send recognized words to the Generative AI API
        response = chat.send_message("'"+str(complete_word + " ', {CONSTRUCT A GRAMATICALLY CORRECT SENTENCE USING THE PROVIDED WORDS, WITHOUT ADDING ADDITIONAL NOUNS OR ADJECTIVES"))
        complete_sentence = response.text
        print("Complete Sentence:", complete_sentence)
        ans = str(complete_sentence)

        if nn in [1, 2, 3, 5]:  # Translation to other languages
            translated_text = translator.translate(ans, dest=lan).text
            print("Translated text: {}".format(translated_text))
            trans_label.config(text=translated_text)  # Updating the sentence label
        else:
            translated_text = ans

        # Default to English if lan is empty
        if not lan:
            lan = 'en'  # Default to English

        # Convert the text to speech
        speech = gTTS(translated_text, lang=lan, slow=False)

        # Save the speech as an mp3 file
        speech_file = "output.mp3"
        speech.save(speech_file)
        song = MP3("output.mp3")
        pygame.mixer.init()
        pygame.mixer.music.load('output.mp3')
        pygame.mixer.music.play()
        time.sleep(song.info.length)
        pygame.quit()

        # Save recognized sentence to a file
        with open('data.txt', 'a') as f:
            f.write(complete_sentence + '\n')

        # Reset captured signs for the next sentence
        captured_signs = []


# Tkinter GUI setup
root = tk.Tk()
root.geometry("1550x800")
root.config(bg="cyan")
root.resizable(0,0)
root.title("ASL and Gemini")

# Canvas for displaying video feed
canvas = tk.Canvas(root, width=1600, height=600)
canvas.pack()

# Label to display the recognized class
class_label = tk.Label(root, text="", font=("Helvetica", 16))
class_label.pack(side=tk.BOTTOM)

# Label to display the complete sentence
sentence_label = tk.Label(root, text="", font=("Helvetica", 16))
sentence_label.pack(side=tk.TOP)

# Label to display the recognized letter
letter_label = tk.Label(root, text="", font=("Helvetica", 16))
letter_label.pack(side=tk.TOP)

# Label to display the complete sentence
trans_label = tk.Label(root, text="", font=("Helvetica", 16))
trans_label.place(x=20,y=670)  # Change the side to TOP

# Label to display the recognized word
word_label = tk.Label(root, text="", font=("Helvetica", 16))
word_label.pack(side=tk.TOP)

# Button to stop capturing and display the complete sentence
stop_button = tk.Button(root, text="Translate Tamil", command=lambda:stop_capture(1,'ta'), font=("Helvetica", 16),width=20)
stop_button.place(x=700,y=250)

stop_button = tk.Button(root, text="Translate Kannada", command=lambda:stop_capture(2,'kn'), font=("Helvetica", 16),width=20)
stop_button.place(x=700,y=310)

stop_button = tk.Button(root, text="Translate Hindi", command=lambda:stop_capture(3,'hi'), font=("Helvetica", 16),width=20)
stop_button.place(x=700,y=370)

stop_button = tk.Button(root, text="Translate Telugu", command=lambda:stop_capture(5,'te'), font=("Helvetica", 16),width=20)
stop_button.place(x=700,y=430)

stop_button = tk.Button(root, text="Create English Sentence", command=lambda:stop_capture(4,'en'), font=("Helvetica", 16),width=20)
stop_button.place(x=700,y=490)

# Button to capture sign
capture_button = tk.Button(root, text="Capture Sign", command=capture_sign, width=20, height=3)
capture_button.place(x=100,y=500)

# Function to add space after the word
def add_space():
    global captured_signs
    if captured_signs and captured_signs[-1] != ' ':
        captured_signs.append(' ')
        print("Space added.")
        word_label.config(text=f"Space Added: {''.join(captured_signs)} (Space Added)")

# Function to remove the last captured sign
def remove_last_sign():
    global captured_signs
    if captured_signs:
        captured_signs.pop()  # Remove the last element from the list
        print("Last sign removed.")

# Button to remove the last recognized letter
backspace_button = tk.Button(root, text="Backspace", command=remove_last_sign, width=20, height=3)
backspace_button.place(x=300,y=500)

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.resize(image, (500, 500))
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                )

            
            without_garbage = []
            clean = []
            data = results.multi_hand_landmarks[0]

            data = str(data).strip().split('\n')
            garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

            for i in data:
                if i not in garbage:
                    without_garbage.append(i)

            for i in without_garbage:
                i = i.strip()
                clean.append(i[2:])

            for i in range(0, len(clean)):
                clean[i] = float(clean[i])

            Class = svm.predict(np.array(clean).reshape(-1, 63))
            Class = Class[0]
            cv2.putText(image, str(Class), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
            if capturing:
                # Append recognized sign to the list
                captured_signs.append(str(Class))

                # Display the recognized class on the label
                class_label.config(text=f"Recognized Class: {Class}")
                print(Class)

                # Reset capturing to False after appending the sign
                capturing = False

            # Update the letter label with the recognized letter
            letter_label.config(text=f"Recognized Letter: {Class}")
            recognized_letter = Class

            if recognized_letter=='Q':
                add_space()

            # Update the word label with the recognized word
            word_label.config(text=f"Recognized Word: {''.join(captured_signs)}")
            recognized_word = ''.join(captured_signs)

        # Display the video feed and GUI
        img = Image.fromarray(image)
        img = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img)

        # Update the sentence label with the complete sentence
        sentence_label.config(text=complete_sentence)  # Updating the sentence label
        root.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    root.mainloop()
