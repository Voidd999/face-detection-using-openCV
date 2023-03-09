from PIL import Image 
from os import walk   
from math import sqrt 

# Defining a function called 'collage' that takes 3 arguments: path to input images,
# path to save the output image grid, and the name of the output file.
def collage(path, grid_path, name):
    
    images = []  # Creating an empty list to store images  
    # Looping through all files in the given path
    for root, dirc, files in walk(path):
        for FileName in files:
            input_image = FileName
            # Opening the image and resizing it to 100x100 pixels using the PIL Image module
            pilimg = Image.open(path+'/'+input_image)
            input_image = pilimg.resize((100,100)) 
            # Adding the resized image to the 'images' list
            images.append(input_image)
    
    n = len(images)                 # Number of images
    rows = int(sqrt(n))             # Number of rows in the grid
    cols = int(n/rows)              # Number of columns in the grid
    
    new_image = Image.new('RGB', (cols*100, rows*100))  # Creating a new blank image of size (cols*100, rows*100)
    
    i = 0
    for y in range(rows):
        if i >= len(images):
            break
        y *= 100
        for x in range(cols):
            x *= 100
            img = images[i]
            new_image.paste(img, (x, y, x+100, y+100))  # Pasting the images on the new_image
            
            i += 1
    
    # Saving the image grid
    try:
     new_image.save(f'{grid_path}/{name}.png')
     print(f'saved a [{rows}x{cols}] grid')
    except Exception as e:
     print(f'Error while saving face_grid: {e}')

