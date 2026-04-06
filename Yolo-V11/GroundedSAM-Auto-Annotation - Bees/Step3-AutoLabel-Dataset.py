from autodistill.detection import CaptionOntology


# Define the Ontology 


ontology = CaptionOntology({
    "a bee" : "bee",
    "a flower" : "flower"
})

# init the Base model 

IMAGE_DIR_PATH = "D:/Data-Sets-Object-Detection/Bees/images"
DATASET_DIR_PATH = "D:/Data-Sets-Object-Detection/Bees/dataset"

from autodistill_grounded_sam import GroundedSAM

base_model = GroundedSAM(ontology=ontology)
dataset = base_model.label(
    input_folder=IMAGE_DIR_PATH,
    output_folder=DATASET_DIR_PATH,
    extension='.png'
)

# This will generate a dataset folder with train and validation images folders and labels folder 
# It will also generate a data.yaml file , ready made for Yolo training 

