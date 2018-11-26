import json
from os.path import join
import glob
import cv2
import numpy as np


def select_correct_data():


    # choose the directory where the json files are saved

    datapath = '/home/erifili/Desktop/project_Deep_learning/Dataset'


    # get the names of the tar files in datapath
    folder_names = glob.glob(join(datapath,'0*'))
    folder_names.sort()



    face_list = []
    right_eye_list = []
    left_eye_list = []
    face_grid_list = []
    y_list = []

    for name in folder_names:

        print(name)
        face = open(join(datapath, name, 'appleFace.json'))
        Face = json.load(face)

        left_eye = open(join(datapath, name, 'appleLeftEye.json'))
        Left_eye = json.load(left_eye)

        right_eye = open(join(datapath, name, 'appleRightEye.json'))
        Right_eye = json.load(right_eye)

        face_grid = open(join(datapath, name, 'faceGrid.json'))
        Face_grid = json.load(face_grid)

        info = open(join(datapath, name, 'info.json'))
        Info = json.load(info)

        frames = open(join(datapath, name, 'frames.json'))
        Frames = json.load(frames)

        y = open(join(datapath, name, 'dotInfo.json'))
        Y = json.load(y)


        for i in range(0, int(Info["TotalFrames"])):

            if Left_eye["IsValid"][i] and Right_eye["IsValid"][i] and Face["IsValid"][i]:

                img = cv2.imread(join(name,'frames',Frames[i]))

                # print(img)



                if int(Face["X"][i]) < 0 or int(Face["Y"][i]) < 0 or \
                        int(Left_eye["X"][i]) < 0 or int(Left_eye["Y"][i]) < 0 or \
                        int(Right_eye["X"][i]) < 0 or int(Right_eye["Y"][i]) < 0:
                    # print("Error with coordinates: {}".format(join(path, dir, "frames", frame)))
                    continue

                    # get face
                tl_x_face = int(Face["X"][i])
                tl_y_face = int(Face["Y"][i])
                w = int(Face["W"][i])
                h = int(Face["H"][i])
                br_x = tl_x_face + w
                br_y = tl_y_face + h
                face = img[tl_y_face:br_y, tl_x_face:br_x]

                # get left eye
                tl_x = tl_x_face + int(Left_eye["X"][i])
                tl_y = tl_y_face + int(Left_eye["Y"][i])
                w = int(Left_eye["W"][i])
                h = int(Left_eye["H"][i])
                br_x = tl_x + w
                br_y = tl_y + h
                left_eye = img[tl_y:br_y, tl_x:br_x]

                # get right eye
                tl_x = tl_x_face + int(Right_eye["X"][i])
                tl_y = tl_y_face + int(Right_eye["Y"][i])
                w = int(Right_eye["W"][i])
                h = int(Right_eye["H"][i])
                br_x = tl_x + w
                br_y = tl_y + h
                right_eye = img[tl_y:br_y, tl_x:br_x]

                # get face grid (in ch, cols, rows convention)
                face_grid = np.zeros(shape=(1, 25, 25), dtype=np.float32)
                tl_x = int(Face_grid["X"][i])
                tl_y = int(Face_grid["Y"][i])
                w = int(Face_grid["W"][i])
                h = int(Face_grid["H"][i])
                br_x = tl_x + w
                br_y = tl_y + h
                face_grid[0, tl_y:br_y, tl_x:br_x] = 1

                # get labels
                y_x = Y["XCam"][i]
                y_y = Y["YCam"][i]

                # resize images
                face = cv2.resize(face, (64, 64))
                left_eye = cv2.resize(left_eye, (64, 64))
                right_eye = cv2.resize(right_eye, (64, 64))


                face_list.append(face)
                left_eye_list.append(left_eye)
                right_eye_list.append(right_eye)
                face_grid_list.append(face_grid)
                y_list.append([y_x, y_y])

    np.savez('test', face=face_list, leftEye=left_eye_list, rightEye= right_eye_list, face_grid = face_grid_list, Labels = y_list )



if __name__ == '__main__':
    select_correct_data()