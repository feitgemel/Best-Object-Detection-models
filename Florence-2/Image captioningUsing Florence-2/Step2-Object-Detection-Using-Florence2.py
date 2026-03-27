import textwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
from PIL import Image ,ImageDraw, ImageFont 
from transformers import AutoProcessor, AutoModelForCausalLM
import cv2 

# Model 
model_id = "microsoft/Florence-2-large"
# model_id = "microsoft/Florence-2-base"

# Init the model 
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             trust_remote_code = True).eval() 

# Init the processor 
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code = True)


# Function that generate the output from a task 

# task - What you want to perform (Object Detectio , Image Captioning , Segmentation )
# Images - The input image 
# text_input -> Optional , part of the task 

def run_florence2(task_promt , image , text_input = None):
    
    if text_input is None :
        prompt = task_promt
    else :
        prompt = task_promt + text_input

    # Run the proccesor and get the tensors 
    inputs = processor(text = prompt , images=image, return_tensors="pt") # The output will be a tensor 

    # generate the output text from the tensors (Inside the inputs )

    generated_ids = model.generate(
        input_ids = inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )

    # Get the data in readable format :
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_promt,
        image_size = (image.width , image.height) )
    
    return parsed_answer # Return the parsed answer 


image = Image.open("Best-Object-Detection-models/Florence-2/Image captioningUsing Florence-2/Parrot.jpg")

# Task 3 - Object Detection  
########################### 
 
task_prompt3  = '<OD>'
results = run_florence2(task_prompt3 , image)
data = results['<OD>']
print(data)



# # Display the result :
# #######################

open_cv_image = np.array(image)
open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

color = (0, 255, 255) # Yellow 
thickness = 3
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 3

# Plot the result 
for bbox , label in zip(data['bboxes'], data['labels']):

    # Unpack the bounding box coordinates 
    x1 , y1, x2, y2 = bbox 
    x1 , y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Draw a rectangle 
    open_cv_image = cv2.rectangle(open_cv_image, (x1, y1), (x2, y2), color, thickness)

    # Annotate the label 
    open_cv_image = cv2.putText(open_cv_image, label, (x1, y1 - 20 ), font, fontScale, color, thickness, cv2.LINE_AA)


scale_percent = 30 # percent of original size
width = int(open_cv_image.shape[1] * scale_percent / 100)
height = int(open_cv_image.shape[0] * scale_percent / 100)
dim = (width, height)

# Resize image 
open_cv_image = cv2.resize(open_cv_image, dim, interpolation = cv2.INTER_AREA)

cv2.imwrite("Best-Object-Detection-models/Florence-2/Image captioningUsing Florence-2/Step2_OD_result.png", open_cv_image)

cv2.imshow("img" , open_cv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

