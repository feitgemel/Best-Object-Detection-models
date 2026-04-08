from autodistill.detection import CaptionOntology


# Define the Ontology
ontology = CaptionOntology({
    "an athletic person running race track": "an athletic person running race track",
    "Athletic runner" : "Athletic runner",
    "Olympic athlete" : "Olympic athlete",})

# Initiate base model and autolabel 

IMAGE_DIR_PATH = "D:/Data-Sets-Object-Detection/Athletic-Runner/images"
DATASET_DIR_PATH = "D:/Data-Sets-Object-Detection/Athletic-Runner/dataset"

BOX_THRESHOLD = 0.5 
TEXT_THRESHOLD = 0.5 

from autodistill_grounding_dino import GroundingDINO

base_model = GroundingDINO(ontology=ontology , box_threshold=BOX_THRESHOLD , text_threshold=TEXT_THRESHOLD)

dataset = base_model.label(
    input_folder=IMAGE_DIR_PATH,
    output_folder=DATASET_DIR_PATH,
    extension='.png'
)



