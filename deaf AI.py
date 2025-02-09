import tkinter as tk
from tkinter import filedialog, messagebox
import speech_recognition as sr
import librosa
import numpy as np
import tensorflow as tf

# Load the pre-trained sound classification model
class_names = ['Siren', 'Dog Bark', 'Doorbell', 'Clapping', 'Background Noise']
model = tf.keras.models.load_model('sound_classification_model.h5')

def classify_sound(audio_path):
    try:
        y, sr = librosa.load(audio_path, duration=2.97, offset=0.5)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs = np.mean(mfccs.T, axis=0)
        mfccs = np.expand_dims(mfccs, axis=0)
        
        predictions = model.predict(mfccs)
        predicted_class = class_names[np.argmax(predictions)]
        return predicted_class
    except Exception as e:
        return f"Error processing audio: {e}"

def transcribe_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            transcription_label.config(text="Listening for speech...")
            audio_data = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio_data)
            transcription_label.config(text=f"Speech Detected: {text}")
        except sr.UnknownValueError:
            transcription_label.config(text="Could not understand the audio.")
        except sr.RequestError as e:
            transcription_label.config(text=f"API error: {e}")
        except Exception as e:
            transcription_label.config(text=f"Error: {e}")

def upload_and_classify():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if not file_path:
        return
    classification_label.config(text="Processing audio...")
    result = classify_sound(file_path)
    classification_label.config(text=f"Sound Detected: {result}")

# GUI Application
app = tk.Tk()
app.title("Deaf Assistance AI")
app.geometry("400x400")

# Title
title_label = tk.Label(app, text="Deaf Assistance AI", font=("Arial", 18, "bold"))
title_label.pack(pady=10)

# Speech-to-Text Section
speech_frame = tk.Frame(app)
speech_frame.pack(pady=10)
speech_button = tk.Button(speech_frame, text="Transcribe Speech", font=("Arial", 12), command=transcribe_speech)
speech_button.pack(pady=5)
transcription_label = tk.Label(speech_frame, text="Speech transcription will appear here.", font=("Arial", 10), wraplength=350)
transcription_label.pack(pady=5)

# Sound Classification Section
sound_frame = tk.Frame(app)
sound_frame.pack(pady=10)
sound_button = tk.Button(sound_frame, text="Classify Sound", font=("Arial", 12), command=upload_and_classify)
sound_button.pack(pady=5)
classification_label = tk.Label(sound_frame, text="Sound classification will appear here.", font=("Arial", 10), wraplength=350)
classification_label.pack(pady=5)

# Exit Button
exit_button = tk.Button(app, text="Exit", font=("Arial", 12), command=app.quit, bg="red", fg="white")
exit_button.pack(pady=20)

app.mainloop()
