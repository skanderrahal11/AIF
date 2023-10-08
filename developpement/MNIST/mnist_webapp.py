import gradio as gr
from PIL import Image
import requests
import io


def recognize_digit(image):
    # Convert to PIL Image necessary if using the API method
    image = Image.fromarray(image.astype('uint8'))
    img_binary = ...
    ...
    ...
    return ....

if __name__=='__main__':

    gr.Interface(fn=recognize_digit, 
                inputs="sketchpad", 
                outputs='label',
                live=True,
                description="Draw a number on the sketchpad to see the model's prediction.",
                ).launch(debug=True, share=True);