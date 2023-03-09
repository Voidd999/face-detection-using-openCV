# Importing necessary libraries and setting the path to the input image
import cv2
from os import walk
from collage import collage

path = '/input'

# Looping through all files in the path
for root, dirc, files in walk(path): 
    for FileName in files:
        # Checking file format
        if FileName.endswith('.png') or FileName.endswith('.jpg'):
            # Saving the image filename and printing it
            input_image = FileName
            print(f'Loaded {input_image}')

# Reading the input image and converting it to grayscale
image = cv2.imread(path+'/'+input_image) 
bw_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Detecting coordinates of faces in the image using pre-trained classifier
trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
coordinates = trained_data.detectMultiScale(bw_img,minNeighbors = 9)

# Saving each detected face as an individual file and drawing a rectangle around the face on the original image
count = 1
path_output = '/saved_faces'
file_name = 'Face_'
for (x,y,w,h) in coordinates: 
    face = image[y:y+h,x+12:(x-12)+w] 
    faces = image[y:y+h,x:x+w] 
    written = cv2.imwrite(f'{path_output}/{file_name}{str(count)}.png', faces) 
    cv2.rectangle(image,(x , y),(x+w ,y+h),(255,255,255),1) 
    count += 1

    # Appending detected face coordinates to a list for no specific use
    faces_detected = []
    faces_detected.append(coordinates)

# Creating a list of all saved faces filenames 
file_names = list()
path = '/saved_faces'
for root, dirc, files in walk(path):
    for FileName in files:
        if FileName.endswith('.png') and FileName.startswith('Face_'):
            file_names.append(FileName) 

# Displaying the original image with detected faces and waiting for user input
while True:  
    cv2.namedWindow('faces',cv2.WINDOW_NORMAL) 
    cv2.resizeWindow('faces',940,610)
    cv2.imshow('faces',image)
    key = cv2.waitKey()
    #waiting for the key Q   
    if key == 81 or key == 113:
        break

# Printing the number of detected and saved faces and their [filenames]
if written:
    print(f'recorded and saved {count-1} faces\n{file_names}')

    # Creating a grid of all saved face images using the collage function
    grid_path = '/face_grid'
    collage(path_output,grid_path,'face_grid')
