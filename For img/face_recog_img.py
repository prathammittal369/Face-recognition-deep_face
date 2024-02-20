import os
import cv2 as cv
import numpy as np
from verify_function import verify
from face_recognition import face_locations


# function for face_loactions
def location_of_all_known_unknown_faces(list_of_faces):

    location_of_all_faces = []

    for all_face in list_of_faces:

        face_location_list = face_locations(all_face)

        if len(face_location_list) > 0:

            [ (top,right,bottom,left) ] = face_location_list
            
            location_of_all_faces.append( (top,right,bottom,left) )
        else:

            # If face_locations is empty, append a placeholder tuple or handle it as needed
            location_of_all_faces.append( (0,0,0,0) )

    return location_of_all_faces


# finding index of faces from list of faces
def find_index(known_unknown_face, list_of_faces):

    for i,face in enumerate(list_of_faces):

        if np.array_equal(known_unknown_face,face):
            return i
        
    return f"Error {known_unknown_face} is not found in find_index function :( ."


# directory of all known faces
dir = r"F:\Python\Vs code programming files\Imgs2\All in one"


# saving all known faces in all_faces
all_faces = [ cv.imread( os.path.join( dir, os.path.join(dir,each_img) ) ) for index,each_img in enumerate( os.listdir(dir) ) if index != 3 ]

all_croped_known_faces = []



# sare known faces ka locations nd saved in :- location_of_all_known_faces
location_of_all_known_faces = location_of_all_known_unknown_faces( all_faces )



# sare known faces ko crop karke nd saved in  :- all_croped_faces
for index , img in enumerate(all_faces):

    all_croped_known_faces.append( img[ location_of_all_known_faces[index][0] : location_of_all_known_faces[index][2], location_of_all_known_faces[index][3] : location_of_all_known_faces[index][1] ] )

# resizing of images
resize_height = 350
resize_width = 350

all_croped_known_faces = [ cv.resize(img, ( resize_width,resize_height ), interpolation = cv.INTER_AREA )  for img in all_croped_known_faces ]


# video streaming
img_collage = cv.imread(r"F:\Python\Vs code programming files\Imgs2\All in one collage\IMG_20240217_162213.jpg")

resize_height = 500
resize_width = 500
img_collage = cv.resize(img_collage, ( resize_width,resize_height ), interpolation = cv.INTER_AREA )

#fetching all faces for a frame
all_face_location_in_frame = face_locations( img_collage )

all_croped_face_in_frame = [ img_collage[ top:bottom, left:right ] for (top,right,bottom,left) in all_face_location_in_frame ]
        
    
for known_face in all_croped_known_faces:

    for unknown_face_in_frame in all_croped_face_in_frame:

        # suppose the width or height of any img is 0 then? socha? uske liye:-
        if not any( face.shape[0] ==  0 for face in [known_face, unknown_face_in_frame]):

            dicto = verify(known_face, unknown_face_in_frame, enforce_detection= False)
            verified , is_true = list( dicto.items() )[0]

            if is_true:
                    
                # got the the index of the known face
                index_of_that_known_face = find_index(known_face,all_croped_known_faces)

                # got the the index of the known face which was detected
                index_whose_face_is_identified = find_index(unknown_face_in_frame, all_croped_face_in_frame)
                #all_croped_face_in_frame.index( unknown_face_in_frame ) aasa isiliye nai kiya kyuki
                #The index method in Python is primarily designed for lists and similar sequences. 
                #It looks for the first occurrence of the specified value and returns its index. When working with NumPy arrays,
                #the index method doesn't work as expected because NumPy arrays are more complex data structures than simple Python lists.


                if index_of_that_known_face == 3:
                    index_of_that_known_face += 1

                name_of_known_face = os.listdir(dir)[index_of_that_known_face] # got the the name of the known face which was detected
                    
                top,right,bottom,left = all_face_location_in_frame[index_whose_face_is_identified]

                cv.rectangle(img_collage, ( left,top), (right,bottom), (0,255,0), 1)
                name_of_known_face = name_of_known_face.split(".")[0]  # just to rid of .jpeg or .png
                cv.putText(img_collage, str(name_of_known_face) , (left,top - 20), 1,1,(255,255,255), 1)

                cv.imshow("Faces", img_collage)


cv.waitKey(0)
