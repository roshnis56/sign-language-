# sign-language- 
This project is a Sign Language Recognition system that translates American Sign Language (ASL) gestures into grammatically correct English sentences using a machine learning model and Google Gemini (Generative AI).

Installation & Requirements:
pip install flask nltk opencv-python mediapipe pillow gtts mutagen pygame googletrans==4.0.0-rc1 google-generativeai
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

Additional Notes
Requires webcam for gesture recognition.
Tested on Python 3.9
Internet connection required for Gemini API and translations.


