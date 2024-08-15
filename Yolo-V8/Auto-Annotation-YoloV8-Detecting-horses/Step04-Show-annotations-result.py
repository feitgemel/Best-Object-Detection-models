import os 
import random 
import matplotlib.pyplot as plt 
import cv2 

# Display 8 random images + annotations

label_names = ["horse race" , "horse" , "horse in a race", "horse racing"]

def get_annotations(original_img , label_file):

    with open(label_file , 'r') as file :
        lines = file.readlines()

    annotations = [] 

    for line in lines:
        values = line.split()
        label = values[0]
        x, y, w, h = map(float, values[1:])
        annotations.append((label, x, y, w, h))

    return annotations
    

def put_annotations_in_image(image, annotations) :
    H, W, _ = image.shape

    for annotation in annotations:
        label , x, y, w, h = annotation
        print(label , x, y, w, h)
        label_name = label_names[int(label)]

        # Convert Yolo coordinates to pixel coordinates 

        x1 = int((x - w / 2) * W)
        y1 = int((y - h / 2) * H)
        x2 = int((x + w / 2) * W)
        y2 = int((y + h / 2) * H)

        # Fraw bounding box

        cv2.rectangle(image , (x1, y1), (x2, y2), (200,200,0), 1)

        # Display label 
        cv2.putText(image , label_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (200,200,0), 2 )

    return image


# Display random images
def display_random_images (folder_path , num_images , label_folder):

    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    selected_images = random.sample(image_files,  num_images)

    for i , image_file in enumerate(selected_images):

        img = cv2.imread(os.path.join(folder_path, image_file))

        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_file_path = os.path.join(label_folder, label_file)

        # read annotations
        annotations_Yolo_format = get_annotations(img , label_file_path)

        # put bounding boxes 
        image_with_annotations = put_annotations_in_image(img , annotations_Yolo_format)
        print(image_with_annotations.shape)
        cv2.imshow("img no. " + str(i) , image_with_annotations)
        cv2.waitKey(0)


images_path = "C:/Data-sets/Horse-race/dataset/train/images"
label_folder = "C:/Data-sets/Horse-race/dataset/train/labels"

num_images = 8

# main Run
display_random_images(images_path , num_images , label_folder)





