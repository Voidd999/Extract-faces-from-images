import cv2
from os import walk
from collage import collage

# Set the path to the input image
path = 'input'

# Loop through all files in the path
for root, dirs, files in walk(path): 
    for FileName in files:
        # Check file format
        if FileName.endswith('.png') or FileName.endswith('.jpg'):
            # Save the image filename and print it
            input_image = FileName
            print(f'Loaded {input_image}')

# Read the input image and convert it to grayscale
image = cv2.imread(path+'/'+input_image) 
bw_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect coordinates of faces in the image using pre-trained classifier
trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
coordinates = trained_data.detectMultiScale(bw_img, minNeighbors=9)

# Save each detected face as an individual file and draw a rectangle around the face on the original image
count = 1
path_output = 'saved_faces'
file_name = 'Face_'
for (x, y, w, h) in coordinates: 
    face = image[y:y+h, x+12:(x-12)+w] 
    faces = image[y:y+h, x:x+w] 
    written = cv2.imwrite(f'{path_output}/{file_name}{str(count)}.png', faces) 
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 1) 
    count += 1
    faces_detected = []
    faces_detected.append(coordinates)

# Create a list of all saved face filenames 
file_names = []
for root, dirs, files in walk(path_output):
    for FileName in files:
        if FileName.endswith('.png') and FileName.startswith('Face_'):
            file_names.append(FileName) 

# Display the original image with detected faces and wait for user input
while True:  
    cv2.namedWindow('faces', cv2.WINDOW_NORMAL) 
    cv2.resizeWindow('faces', 940, 610)
    cv2.imshow('faces', image)
    key = cv2.waitKey()
    # Wait for the key 'Q' (81 or 113)
    if key == 81 or key == 113:
        break

# Print the number of detected and saved faces and their filenames
if written:
    print(f'Recorded and saved {count-1} faces:\n{file_names}')

    # Create a grid of all saved face images using the collage function
    grid_path = 'face_grid'
    collage(path_output, grid_path, 'face_grid')
