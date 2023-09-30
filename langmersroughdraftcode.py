import pytesseract
from PIL import Image
from translate import Translator
from gtts import gTTS
from playsound import playsound
from langdetect import detect
from ctypes import cdll
import os
import random

lib = cdll.LoadLibrary('./hardware.so')


class LangMers:
    """ 
    A class for the LangMers system. 
    
    Attributes:
    ----------
    image_path : str
        The path to the image. Default value is '/path/to/image.jpg'.
    translator : Translator
        The Translator object for translating text.

    Methods:
    -------
    capture_image():
        Captures the image.
    get_geolocation_data():
        Gets the geolocation data from the libraby function.
    extract_text_from_image():
        Extracts text from the input image using pyTesseract.
    recognize_language(text: str) -> str:
        Recognizes language of the input text.
    translate_text(text: str, dest_language: str = 'fr') -> str:
        Translates the input text into the destination language. Default destination language is french.
    word_generator(text: str) -> Generator:
        Yields word in the text.
    language_processing(text: str, dest_language: str = 'fr') -> str:
        Process the language and returns the translated text.
    build_pronunciation_feedback(text: str, detected_language: str):
        Saves the pronunciation of the input text in an audio file and plays it.
    run():
        Main function to run the LangMers system.
    """

    def __init__(self, image_path='/path/to/image.jpg'):
        self.image_path = image_path
        self.translator = Translator(service_urls=['translate.google.com'])

    def capture_image(self):
        try:
            lib.capture_image()
        except Exception as e:
            print(f"Unable to capture image: {str(e)}")

    def get_geolocation_data(self):
        try:
            lib.get_geolocation_data()
        except Exception as e:
            print(f"Unable to get geolocation data: {str(e)}")

    def extract_text_from_image(self):
        try:
            img = Image.open(self.image_path)
            text = pytesseract.image_to_string(img, lang="eng")
            return text
        except Exception as e:
            print(f"Unable to open file: {str(e)}")

    def recognize_language(self, text):
        try:
            return detect(text)
        except:
            return "Unable to detect language"

    def translate_text(self, text, dest_language='fr'):
        try:
            result = self.translator.translate(text, dest_language)
            return result.text
        except Exception as e:
            return f"Couldn't translate due to : {str(e)}"

    @staticmethod
    def word_generator(text):
        for word in text.split():
            yield word

    def language_processing(self, text, dest_language='fr'):
        result = self.translate_text(text, dest_language)
        return result

    def build_pronunciation_feedback(self, text, detected_language):
        try:
            tts = gTTS(text=text, lang=detected_language)
            audio_path = 'text_audio.mp3'
            tts.save(audio_path)
            playsound(audio_path)
        except Exception as e:
            print(f'Error in build pronunciation feedback: {str(e)}')

    def run(self):
        self.capture_image()
        self.get_geolocation_data()
        text = self.extract_text_from_image()
        detected_language = self.recognize_language(text)
        self.build_pronunciation_feedback(text, detected_language)


def main():
    langMers = LangMers()
    langMers.run()


if __name__ == "__main__":
    main()