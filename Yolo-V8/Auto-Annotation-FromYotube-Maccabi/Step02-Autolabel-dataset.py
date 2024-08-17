from autodistill.detection import CaptionOntology

# define ontology 

ontology = CaptionOntology({
    "basketball player with blue shirt": "Maccabi player",
    "basketball player with white shirt": "Real Madrid player",
    "Baskball orange ball": "ball",
    "person with black or orange shirt" : "referee",
})

IMAGE_DIR_PATH = "C:/Data-sets/Mac-Real/images"
DATASET_DIR_PTH = "C:/Data-sets/Mac-Real/dataset"

BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.3 

from autodistill_grounding_dino import GroundingDINO

base_model = GroundingDINO(ontology=ontology , box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD)

dataset = base_model.label(input_folder=IMAGE_DIR_PATH , extension=".png" , output_folder=DATASET_DIR_PTH)

