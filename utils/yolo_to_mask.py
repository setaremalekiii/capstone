from ultralytics.data.converter import yolo_bbox2segment

# you need to pip install ultralytics package for this to work
# This takes hours to run so only do it on a small number of images ~300 took at least 1.5 hours 
yolo_bbox2segment(
    # change this accordingly 
    im_dir="norm/images/train",
    save_dir=None,          # creates labels-segment/ next to images
    sam_model="sam_b.pt"    # path to SAM model
)

