# Yolo optimized

# Creating Synthetic Image Datasets
This tool helps create synthetic data for object detection modeling. Given
a folder of background images and object images, this tool iterates through each
background and superimposes objects within the frame in random locations,
automatically annotating as it goes. The tool also resizes the icons to help the
model generalize better to the real world.

# GAMMA IDEA - NOT TESTED TRAINING METHOD -


## Setup
Clone this repo. Then create and activate the conda environment provided:
```bash
$ conda env create -f environment.yml
$ conda activate images
```

Place background images in the `Backgrounds/` subfolder and objects in
the `Objects/` subfolder.

## Create
Run the `create.py` script to generate hundreds/thousands of synthetic training
images for object detection models.

```bash
$ python create.py
```

Output images will be placed in the `TrainingData/` subfolder once done.

### Args
These are the available entrypoint arguments that you can supply at runtime. More will be added in the future.

- `--backgrounds`: Path to folder of background images.
- `--objects`    : Path to folder of object images.
- `--output`     : Path to folder of output images.
- `--groups`     : Whether or not to place groups of objects together.
- `--annotate`   : Whether or not to create and save annotations for the new images.
- `--sframe`     : Whether or not to create a Turi Create SFrame for modeling.
- `--mutate`     : Perform mutatuons to objects (rotation, brightness, shapness, contrast)
- `--outsync`    : Write a sync file that can be used to generate same generated data given same backgrounds and objects, ex. revising generated data using labimg offline without downloading results from colab, only downloading sync file
- `--insync`     : Read the sync file to generate same data generated from outsync method
- `--doclasses`  : Generate the classes files
