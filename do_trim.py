import os
import os.path

for file in os.listdir("./Objects"):
    if file.lower().__contains__('.png') or file.lower().__contains__('.jp'):
        print(file)
        x = "./Objects/"+file
        os.system("convert -trim "+x+" "+ x)
