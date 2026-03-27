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


image = Image.open("Best-Object-Detection-models/Florence-2/Image captioningUsing Florence-2/Book-Page.jpg")

# Task 5 - OCR with Region  
################################################# 
 
task_prompt5  = "<OCR_WITH_REGION>"
results = run_florence2(task_prompt5 , image)
data = results["<OCR_WITH_REGION>"]
print(data)



# # Display the result :
# #######################

open_cv_image = np.array(image)
open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

color = (0, 0, 255) 
thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5

# Plot the result 
for bbox , label in zip(data['quad_boxes'], data['labels']):

    # Unpack the bounding box coordinates 
    polygon = np.array(bbox, dtype=np.int32).reshape((-1,1,2)) # Ensure the shape is (M,1,2)
    # Draw a polygon 
    cv2.polylines(
        open_cv_image, 
        [polygon], # List of Polygons
        isClosed=True, 
        color=(0,255,255), # Yellow
        thickness=thickness  
    )

    # Annotate the label , and add the text in red color 
    x, y = polygon[0][0] # Get the top-left corner of the polygon for the label 

    open_cv_image = cv2.putText(open_cv_image, label, (x, y - 5 ), font, fontScale, color, 1, cv2.LINE_AA)


# scale_percent = 30 # percent of original size
# width = int(open_cv_image.shape[1] * scale_percent / 100)
# height = int(open_cv_image.shape[0] * scale_percent / 100)
# dim = (width, height)

# # Resize image 
# open_cv_image = cv2.resize(open_cv_image, dim, interpolation = cv2.INTER_AREA)
cv2.imwrite("Best-Object-Detection-models/Florence-2/Image captioningUsing Florence-2/Step4_OCR_result.png", open_cv_image)
cv2.imshow("img" , open_cv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

