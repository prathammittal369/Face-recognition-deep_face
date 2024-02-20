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


class FaceRecognizer:
    def __init__(self, known_faces_dir):
        self.known_faces_dir = known_faces_dir
        self.all_known_faces = self.load_all_known_faces()
        self.all_croped_known_faces = self.crop_known_faces()

    def load_all_known_faces(self):
        return [cv.imread(os.path.join(self.known_faces_dir, each_img)) for index, each_img in enumerate(os.listdir(self.known_faces_dir)) if index != 3]

    def crop_known_faces(self):
        location_of_all_known_faces = location_of_all_known_unknown_faces(self.all_known_faces)
        return [img[loc[0]:loc[2], loc[3]:loc[1]] for img, loc in zip(self.all_known_faces, location_of_all_known_faces)]

class VideoFaceRecognition:
    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.video = cv.VideoCapture(0)

    def run(self):
        is_true = True
        while is_true:
            Break = True
            is_true, frame = self.video.read()
            if is_true == False:
                break

            all_face_location_in_frame = face_locations(frame)
            all_croped_face_in_frame = [frame[top:bottom, left:right] for (top, right, bottom, left) in all_face_location_in_frame]

            for known_face in self.recognizer.all_croped_known_faces:
                for unknown_face_in_frame in all_croped_face_in_frame:
                    if not any(face.shape[0] == 0 for face in [known_face, unknown_face_in_frame]):
                        dicto = verify(known_face, unknown_face_in_frame, enforce_detection=False)
                        verified, is_true = list(dicto.items())[0]

                        if is_true:
                            index_of_known_face = find_index(known_face, self.recognizer.all_croped_known_faces)
                            index_whose_face_is_identified = find_index(unknown_face_in_frame, all_croped_face_in_frame)
                            # all_croped_face_in_frame.index( unknown_face_in_frame ) aasa isiliye nai kiya kyuki
                            #The index method in Python is primarily designed for lists and similar sequences. 
                            #It looks for the first occurrence of the specified value and returns its index. When working with NumPy arrays,
                            #the index method doesn't work as expected because NumPy arrays are more complex data structures than simple Python lists.


                            if index_of_known_face == 3:
                                index_of_known_face += 1

                            name_of_known_face = os.listdir(self.recognizer.known_faces_dir)[index_of_known_face]

                            top, right, bottom, left = all_face_location_in_frame[index_whose_face_is_identified]
                            cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
                            cv.putText(frame, str(name_of_known_face), (left, top - 20), 1, 1, (0, 0, 255), 1)

                            cv.imshow("Faces", frame)

                            if cv.waitKey(10) & 0XFF == ord(' '):
                                Break = False
                                break
                if Break == False:
                    break

if __name__ == "__main__":
    known_faces_directory = r"F:\Python\Vs code programming files\Imgs2\All in one"
    recognizer = FaceRecognizer(known_faces_directory)
    video_recognition = VideoFaceRecognition(recognizer)
    video_recognition.run()

    video_recognition.video.release()
    cv.destroyAllWindows()
