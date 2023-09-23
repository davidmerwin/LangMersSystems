import cv2
import numpy as np
import requests
import json

def get_word_details(word):
    # fake API call to demonstrate the process
    response = requests.get(f"https://api.wordnik.com/v4/word.json/{word}")
    word_info = response.json()
    print(f"Looked up details for the word: {word_info}")

def scan_image():
    # Load an color image 
    img = cv2.imread('object.png')

    # Define the list of boundaries
    boundaries = [([17, 15, 100], [50, 56, 200])]
    
    # loop over the boundaries
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
        
        # find the colors within the specified boundaries and apply the mask
        mask = cv2.inRange(img, lower, upper)
        output = cv2.bitwise_and(img, img, mask = mask)
    
        # process output for word recognition, for simplicity let's say we recognized word 'table'
        if output.any(): 
            recognized_word = "table" 

        get_word_details(recognized_word)

if __name__ == "__main__":
    scan_image()
