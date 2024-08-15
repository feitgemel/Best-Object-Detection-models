from autodistill.detection import CaptionOntology

# define ontology 
ontology = CaptionOntology({
    "horse race": "horse race",
    "horse":"horse",
    "horse in a race" : "horse in a race",
    "horse racing" : "horse racing",})

IMAGE_DIR_PATH = "C:/Data-sets/Horse-race/Source-Data/All-images"
DATASET_DIR_PATH = "C:/Data-sets/Horse-race/dataset"

BOX_THRESHOLD = 0.6 
TEXT_THRESHOLD = 0.50 

from autodistill_grounding_dino import GroundingDINO 

base_model = GroundingDINO(ontology=ontology,
                           box_threshold=BOX_THRESHOLD,
                           text_threshold=TEXT_THRESHOLD)



dataset = base_model.label(
    input_folder=IMAGE_DIR_PATH,
    extension=".png",
    output_folder=DATASET_DIR_PATH)