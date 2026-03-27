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



# Task 1 - Image Captioning 
########################### 
# 
task_prompt  = '<CAPTION>'
image = Image.open("Best-Object-Detection-models/Florence-2/Image captioningUsing Florence-2/Parrot.jpg")

result = run_florence2(task_prompt , image)
caption1 = list(result.values() )[0]

print("**********************************************************")
print("Task 1 - Caption : ")
print(caption1)
print("**********************************************************")



# Task 2 - More detailed caption 
#################################

task_prompt2 = '<MORE_DETAILED_CAPTION>'
result = run_florence2(task_prompt2 , image)
long_caption1 = list(result.values() )[0]

print("**********************************************************")
print("Task 2 - MORE_DETAILED_CAPTION : ")
print('\n'.join(textwrap.wrap(long_caption1))) 
print("**********************************************************")




# Display the result :
#######################

open_cv_image = np.array(image)
open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

scale_percent = 30 # percent of original size
width = int(open_cv_image.shape[1] * scale_percent / 100)
height = int(open_cv_image.shape[0] * scale_percent / 100)
dim = (width, height)

# Resize image 
open_cv_image = cv2.resize(open_cv_image, dim, interpolation = cv2.INTER_AREA)

# Add the text to the image 
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
blue = (255, 0, 0)
thickness = 2

open_cv2_image = cv2.putText(open_cv_image, caption1, (50,50), font, fontScale, blue, thickness, cv2.LINE_AA)

cv2.imwrite("Best-Object-Detection-models/Florence-2/Image captioningUsing Florence-2/Step1_Caption_result.png", open_cv_image)

cv2.imshow("img" , open_cv2_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

