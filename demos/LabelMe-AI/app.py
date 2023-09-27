### 1) Import necessary libraries

import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from timeit import default_timer as timer
import gradio as gr

### 2) Define class names and mappings

class_map_s2i = {
    "person": 0, "car": 1, "building": 2, "window": 3, "tree": 4,
    "sign": 5, "door": 6, "bookshelf": 7, "chair": 8, "table": 9,
    "keyboard": 10, "head": 11, "I don't know": 12
}

class_names = list(class_map_s2i.keys())

### 3) Define model

def load_mobilenet_v3_small(model_path="pretrained_MobileNet_V3_small.pth"):
    # Create an EffNetB2 feature extractor
    def create_mobilenet_v3_small():
        # Set the manual seeds
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        # Get the length of class_names (one output unit for each class)
        class_names = class_map_s2i.keys()
        output_shape = len(class_names)

        # Get the base model with pretrained weights and send to target device
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
        model = torchvision.models.mobilenet_v3_small(weights=weights).to('cpu')

        # Change the classifier head
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=576, out_features=output_shape)
        ).to('cpu')

        # Give the model a name
        model_name = "mobilenet_v3_small"
        print(f"[INFO] Created new {model_name} model.")
        return model

    saved_model = create_mobilenet_v3_small().to('cpu')
    saved_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    saved_model.eval()
    return saved_model

### 4) Define the predict function

def predict_image(image, model=load_mobilenet_v3_small()):

    # Start the timer
    start_time = timer()

    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(model(image), dim=1)
    
    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    
    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time

### 5) Gradio App

example_list = ['examples/car.jpg', 'examples/person.jpg', 'examples/chair.jpg', 'examples/window.jpg']

# Create title, description and article strings
title = "LabelMe Classifier \n\n 🙍🚗🏠🪟🌳🛑🚪🪑⌨🙂"
description = "A MobileNetV3 feature extractor computer vision model to classify images trained on the [LabelMe 12 50K dataset](https://www.kaggle.com/datasets/dschettler8845/labelme-12-50k)."
article = "Created at [LabelMe-Classification-AI](https://github.com/AlbertHunduza/LabelMe-Classification-AI). \n\n *NB: this model is not perfect and may make mistakes as it was trained on a very limited, challenging and noisy dataset. It can only classify images in the 12 classes - **person, car, building, window, tree, sign, door, bookshelf, chair, table, keyboard and head.***"

# Create the Gradio demo
demo = gr.Interface(fn=predict_image, # mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=1, label="🤔..."), # what are the outputs?
                             gr.Number(label="Prediction time (s)")], # our fn has two outputs, therefore we have two outputs
                    examples=example_list, 
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch(debug=False, # print errors locally?
            share=True) # generate a publically shareable URL?