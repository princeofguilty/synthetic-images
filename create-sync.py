import os
import json
import argparse
import numpy as np
import random
import math
from PIL import Image, ImageEnhance
from numpy.lib.shape_base import split
import cv2
import imutils
import glob


skip = "n"

# Entrypoint Args
parser = argparse.ArgumentParser(description='Create synthetic training data for object detection algorithms.')
parser.add_argument("-bkg", "--backgrounds", type=str, default="Backgrounds/",
                    help="Path to background images folder.")
parser.add_argument("-obj", "--objects", type=str, default="Objects/",
                    help="Path to object images folder.")
parser.add_argument("-o", "--output", type=str, default="TrainingImages/",
                    help="Path to output images folder.")
parser.add_argument("-ann", "--annotate", type=bool, default=False,
                    help="Include annotations in the data augmentation steps?")
parser.add_argument("-s", "--sframe", type=bool, default=False,
                    help="Convert dataset to an sframe?")
parser.add_argument("-g", "--groups", type=bool, default=False,
                    help="Include groups of objects in training set?")
parser.add_argument("-mut", "--mutate", type=bool, default=True,
                    help="Perform mutatuons to objects (rotation, brightness, shapness, contrast)")
parser.add_argument("-dc", "--doclasses", type=bool, default=False,
                    help="define classid index")
parser.add_argument("-osync", "--outsync", type=bool, default=True,
                    help="prints each background with each object position for revising offline")                    
parser.add_argument("-isync", "--insync", type=bool, default=False,
                    help="take background and objects positions from sync file")    

# sync : background.png object.png size object_x_pos object_y_pos mutate_x mutate_y

args = parser.parse_args()



# Prepare data creation pipeline
base_bkgs_path = args.backgrounds
bkg_images = [f for f in os.listdir(base_bkgs_path) if not f.startswith(".")]
objs_path = args.objects
obj_images = [f for f in os.listdir(objs_path) if not f.startswith(".")]
# sizes = [0.4, 0.6, 0.8, 1, 1.2] # different obj sizes to use TODO make configurable
sizes = [1,1] # different obj sizes to use TODO make configurable
count_per_size = 4 # number of locations for each obj size TODO make configurable
annotations = [] # store annots here
output_images = args.output
n = 1


# Helper functions
def get_obj_positions(obj, bkg, count=1):
    obj_w, obj_h = [], []
    x_positions, y_positions = [], []
    bkg_w, bkg_h = bkg.size
    # Rescale our obj to have a couple different sizes
    obj_sizes = [tuple([int(s*x) for x in obj.size]) for s in sizes]
    for w, h in obj_sizes:
        obj_w.extend([w]*count)
        obj_h.extend([h]*count)
        max_x, max_y = bkg_w-w, bkg_h-h
        x_positions.extend(list(np.random.randint(0, max_x, count)))
        y_positions.extend(list(np.random.randint(0, max_y, count)))
    return obj_h, obj_w, x_positions, y_positions


def get_box(obj_w, obj_h, max_x, max_y):
    x1, y1 = np.random.randint(0, max_x, 1), np.random.randint(0, max_y, 1)
    x2, y2 = x1 + obj_w, y1 + obj_h
    return [x1[0], y1[0], x2[0], y2[0]]


# check if two boxes intersect
def intersects(box, new_box):
    box_x1, box_y1, box_x2, box_y2 = box
    x1, y1, x2, y2 = new_box
    return not (box_x2 < x1 or box_x1 > x2 or box_y1 > y2 or box_y2 < y1)


def get_group_obj_positions(obj_group, bkg):
    bkg_w, bkg_h = bkg.size
    boxes = []
    objs = [Image.open(objs_path + obj_images[i]) for i in obj_group]
    obj_sizes = [tuple([int(0.6*x) for x in i.size]) for i in objs]
    for w, h in obj_sizes:
        # set background image boundaries
        max_x, max_y = bkg_w-w, bkg_h-h
        # get new box coordinates for the obj on the bkg
        while True:
            new_box = get_box(w, h, max_x, max_y)
            for box in boxes:
                res = intersects(box, new_box)
                if res:
                    break

            else:
                break  # only executed if the inner loop did NOT break
            #print("retrying a new obj box")
            continue  # only executed if the inner loop DID break
        # append our new box
        boxes.append(new_box)
    return obj_sizes, boxes
    
def mutate_image(img, ang=-1):
    # resize image for random value
    resize_rate = random.choice(sizes)
    img = img.resize([int(img.width*resize_rate), int(img.height*resize_rate)], Image.ANTIALIAS)

    # rotate image for random andle and generate exclusion mask 
    if ang == -1:
        rotate_angle = random.randint(0,360)
    else:
        rotate_angle = ang
    img = img.rotate(rotate_angle, expand=True)


    # perform some enhancements on image
    enhancers = [ImageEnhance.Brightness, ImageEnhance.Color, ImageEnhance.Contrast, ImageEnhance.Sharpness]
    enhancers_count = random.randint(0,3)
    for i in range(0,enhancers_count):
        enhancer = random.choice(enhancers)
        enhancers.remove(enhancer)
        img = enhancer(img).enhance(random.uniform(0.5,1.5))
    img.save("tmp.png")
    os.system("convert -trim tmp.png tmp.png")
    img = Image.open("tmp.png")

    return img, rotate_angle

classesFile = set([])

def doClassesFile(dir):
    global classesFile
    for i in glob.glob(os.path.join(dir, '*.png')):
        if (i.__contains__('_')):
            classesFile.add(i.split('_')[1].split('.')[0])

    classesFile = '\n'.join(classesFile)

    with open("./TrainingImages/classes.txt", "w") as f:
        f.write(classesFile)
        


if __name__ == "__main__":

    if(args.outsync and args.insync):
        print("cant insync and outsync at same time")
        exit(1)

    # Make synthetic training data
    print("Making synthetic images.", flush=True)
    doClassesFile(objs_path)
    if (args.outsync):
        if os.path.exists("sync.txt"):
            os.remove("sync.txt")
        osync = open("sync.txt", "a")

    if(args.insync):
        isync = open("sync.txt", 'r').read()
        isync_lines = isync.split('\n')
        condition = isync_lines
    else : 
        condition = bkg_images

    for line in condition:
        # Load the background image
        if(line==""):
            break
        if (args.insync):
            line = line.split(' ')
            bkg_path = line[0]
        else:
            bkg_path = base_bkgs_path + line
        try :
            bkg_img = Image.open(bkg_path)
        except Exception as e:
            print(e)
            continue
        bkg_x, bkg_y = bkg_img.size
        
        # Do single objs first

        if (args.insync):
            obj_images = str(line[1])

        for i in obj_images:
            try :
                if not args.insync:
                # Load the single obj
                    if not (i.__contains__('.png') or i.__contains__('.PNG') or i.__contains__('jpg') or i.__contains__('.JPG')):
                        continue
                    i_path = objs_path + i
                    print(i_path)
                    obj_img = Image.open(i_path)
                    

                    # Get an array of random obj positions (from top-left corner)
                    obj_h, obj_w, x_pos, y_pos = get_obj_positions(obj=obj_img, bkg=bkg_img, count=count_per_size)            
                else :
                    obj_img = Image.open(str(line[1]))
                    

                # Create synthetic images based on positions
                if not args.insync:
                    for h, w, x, y in zip(obj_h, obj_w, x_pos, y_pos):
                        # Copy background
                        if(args.outsync):
                            osync.write(str(bkg_path) + " " +str(i_path) + " ")
                        bkg_w_obj = bkg_img.copy()
                        if(args.outsync):
                            osync.write( str(h) + " " + str(w) + " " + str(x) + " " + str(y) + " ")
                        if args.mutate:
                            new_obj, ang = mutate_image(obj_img)

                            # osync.write(str())
                            # Paste on the obj
                            bkg_w_obj.paste(new_obj, (x, y), new_obj)
                        else:
                            # Adjust obj size
                            new_obj = obj_img.resize(size=(w, h))
                            # Paste on the obj
                            bkg_w_obj.paste(new_obj, (x, y), new_obj)
                        output_fp = output_images + str(n) + ".png"
                        classid = i.split('_')[1].split('.')[0]
                        noExtName = output_fp.split('.')[0]
                        newFileName = noExtName+".txt"
                        if args.mutate:
                            with open(newFileName, "w") as f:
                                # new_obj.show()
                                # x1,y1,w1,h1 = getContourDims(cv2.cvtColor(np.array(new_obj), cv2.COLOR_RGB2BGR))
                                x1,y1,w1,h1 = 0,0,new_obj.size[0] ,new_obj.size[1]
                                osync.write(str(h1) +" "+ str(w1) +" "+ str(x1) +" "+ str(y1) + " " + str(ang))
                                # x1,y1,w1,h1 = getContourDims(cv2.cvtColor(np.array(new_obj), cv2.COLOR_RGB2BGR))
                                data = str(classid) + " " + str(((x+x1+(0.5*w1))/bkg_x).round(6)) + " " + str(((y+y1+(0.5*h1))/bkg_y).round(6)) + " " + str(round((w1/bkg_x),6)) + " " + str(round((h1/bkg_y),6))
                                print(str(data) + " " + newFileName)
                                f.write(data)
                        else:
                            with open(newFileName, "w") as f:
                                data = str(classid) + " " + str(((x+(0.5*w))/bkg_x).round(6)) + " " + str(((y+(0.5*h))/bkg_y).round(6)) + " " + str(round((w/bkg_x),6)) + " " + str(round((h/bkg_y),6))
                                print(data + " " + newFileName)
                                f.write(data)
                        if(args.outsync):
                            osync.write("\n")
                        
                        n += 1
                        # Save the image
                        bkg_w_obj.save(fp=output_fp, format="png")
                else:
                    h,w,x,y = int(line[2]), int(line[3]), int(line[4]), int(line[5])
                    if(args.insync):
                        h, w, x, y = int(line[2]), int(line[3]), int(line[4]), int(line[5])
                    # Copy background
                    if(args.outsync):
                        osync.write(str(bkg_path) + " " +str(i_path) + " ")
                    bkg_w_obj = bkg_img.copy()
                    if(args.outsync):
                        osync.write( str(h) + " " + str(w) + " " + str(x) + " " + str(y) + " ")
                    if args.mutate:
                        ang = int(line[10])
                        new_obj, _ = mutate_image(obj_img, ang)
                        # osync.write(str())
                        # Paste on the obj
                        bkg_w_obj.paste(new_obj, (x, y), new_obj)
                    else:
                        # Adjust obj size
                        new_obj = obj_img.resize(size=(w, h))
                        # Paste on the obj
                        bkg_w_obj.paste(new_obj, (x, y), new_obj)
                    output_fp = output_images + str(n) + ".png"
                    if args.insync:
                        classid = line[1].split('.')[0].split('_')[1]
                    else:
                        classid = i.split('_')[1].split('.')[0]
                    noExtName = output_fp.split('.')[0]
                    newFileName = noExtName+".txt"
                    if args.mutate:
                        with open(newFileName, "w") as f:
                            # new_obj.show()
                            if args.insync:
                                h1,w1,x1,y1 = int(line[6]), int(line[7]), int(line[8]), int(line[9])
                            # x1,y1,w1,h1 = getContourDims(cv2.cvtColor(np.array(new_obj), cv2.COLOR_RGB2BGR))
                                data = str(classid) + " " + str(round(((x+x1+(0.5*w1)))/bkg_x, 6)) + " " + str(round(((y+y1+(0.5*h1)))/bkg_y,6)) + " " + str(round((w1/bkg_x),6)) + " " + str(round((h1/bkg_y),6))
                                print(str(data) + " " + newFileName)
                                f.write(data)
                                xk = data.split(' ')
                                for ix in xk[:-1]:
                                    if float(ix) >1:
                                        pass    
                    else:
                        with open(newFileName, "w") as f:
                            data = str(classid) + " " + str(((x+(0.5*w))/bkg_x).round(6)) + " " + str(((y+(0.5*h))/bkg_y).round(6)) + " " + str(round((w/bkg_x),6)) + " " + str(round((h/bkg_y),6))
                            print(data + " " + newFileName)
                            f.write(data)
                    if(args.outsync):
                        osync.write("\n")
                    bkg_w_obj.save(fp=output_fp, format="png")
                    n += 1


                    if args.annotate:
                        # Make annotation
                        ann = [{'coordinates': {'height': h, 'width': w, 'x': x+(0.5*w), 'y': y+(0.5*h)}, 'label': i.split(".png")[0]}]
                        # Save the annotation data
                        annotations.append({
                            "path": output_fp,
                            "annotations": ann
                        })
                
                # print(n)
                

                if(args.insync):
                    break
                
            except Exception as e:
                print(e)
                if skip=="s":
                    skip=input("continue")
                continue

        if args.groups:
            # 24 Groupings of 2-4 objs together on a single background
            groups = [np.random.randint(0, len(obj_images) -1, np.random.randint(2, 5, 1)) for r in range(2*len(obj_images))]
            # For each group of objs
            for group in groups:
                # Get sizes and positions
                ann = []
                obj_sizes, boxes = get_group_obj_positions(group, bkg_img)
                bkg_w_obj = bkg_img.copy()

                # For each obj in the group
                for i, size, box in zip(group, obj_sizes, boxes):
                    # Get the obj
                    obj = Image.open(objs_path + obj_images[i])
                    obj_w, obj_h = size
                    # Resize it as needed
                    new_obj = obj.resize((obj_w, obj_h))
                    x_pos, y_pos = box[:2]
                    if args.annotate:
                        # Add obj annotations
                        annot = {
                                'coordinates': {
                                    'height': obj_h,
                                    'width': obj_w,
                                    'x': int(x_pos+(0.5*obj_w)),
                                    'y': int(y_pos+(0.5*obj_h))
                                },
                                'label': obj_images[i].split(".png")[0]
                            }
                        ann.append(annot)
                    # Paste the obj to the background
                    bkg_w_obj.paste(new_obj, (x_pos, y_pos), new_obj)

                output_fp = output_images + str(n) + ".png"
                # Save image
                bkg_w_obj.save(fp=output_fp, format="png")
                if args.annotate:
                    # Save annotation data
                    annotations.append({
                        "path": output_fp,
                        "annotations": ann
                    })
                #print(n)
                n += 1
    
    if args.annotate:
        print("Saving out Annotations", flush=True)
        # Save annotations
        with open("annotations.json", "w") as f:
            f.write(json.dumps(annotations))

    if args.sframe:
        print("Saving out SFrame", flush=True)
        # Write out data to an sframe for turicreate training
        import turicreate as tc
        # Load images and annotations to sframes
        images = tc.load_images(output_images).sort("path")
        annots = tc.SArray(annotations).unpack(column_name_prefix=None).sort("path")
        # Join
        images = images.join(annots, how='left', on='path')
        # Save out sframe
        images[['image', 'path', 'annotations']].save("training_data.sframe")

    # total_images = len([f for f in os.listdir(output_images) if not f.startswith(".")])
    print("Done! Created {} synthetic training images.".format(n-1), flush=True)
