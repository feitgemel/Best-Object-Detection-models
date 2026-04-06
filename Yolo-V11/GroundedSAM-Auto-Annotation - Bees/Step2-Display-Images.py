import random 
import os 
import cv2 
import matplotlib.pyplot as plt

# Display random images 

# a funcation to load and display 9 random images 

def display_random_images(folder_path):

    # Get all the files in the folder 
    all_files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]

    # Filter only images file (common formats)
    image_files = [file for file in all_files if file.lower().endswith(('.png', '.jpg','.jpeg', '.gif' , '.bmp', '.tiff')) ]

    # Select 9 random images 
    selected_images = random.sample(image_files, min(9, len(image_files)))

    # Create a 3X3 grid for diaply images 

    fig , axes = plt.subplots(3,3, figsize=(10,10))
    fig.subplots_adjust(hspace=0.3 , wspace=0.3)

    # Loop through the grid and pupulate the images 

    for ax , img_name in zip(axes.ravel() , selected_images) :
        img_path = os.path.join(folder_path , img_name) 
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(img_name, fontsize=8 )

    # Hide any unused subplots if fewer the 9 images 
    for ax in axes.ravel()[len(selected_images):] :
        fig.axis('off')

    plt.show()


# Call the funcion 
IMAGE_DIR_PATH = "D:/Data-Sets-Object-Detection/Bees/images"
display_random_images(IMAGE_DIR_PATH)



    