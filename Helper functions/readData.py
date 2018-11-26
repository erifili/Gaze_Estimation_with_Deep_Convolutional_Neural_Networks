import numpy as np
from skimage.color import rgb2gray

'''
    call readData() to import train and validation set 
    and prepare the image data to insert the CNN.
    set mode to 'gray' to return an [images x X x Y x 1] 
    instead of [images x X x Y x 3] array. 
'''

def readData():
    mode = ''
    data = np.load('eye_tracker_train_and_val.npz')

    train_face = data['train_face']
    train_left_eye = data['train_eye_left']
    train_right_eye = data['train_eye_right']
    train_face_grid = data['train_face_mask']
    train_labels = data['train_y']
    val_face = data['val_face']
    val_left_eye = data['val_eye_left']
    val_right_eye = data['val_eye_right']
    val_face_grid = data['val_face_mask']
    val_labels = data['train_y']

    train_data = [train_face, train_left_eye, train_right_eye, train_face_grid, train_labels]
    val_data = [val_face, val_left_eye, val_right_eye, val_face_grid, val_labels]

    train_data = prepare_data(train_data, mode)
    val_data = prepare_data(val_data, mode)

    return train_data, val_data


def prepare_data(data, mode):
    face, left_eye, right_eye, face_grid, labels = data
    face = normalize(face, mode)
    left_eye = normalize(left_eye, mode)
    right_eye = normalize(right_eye, mode)
    face_grid = face_grid.astype('float32').reshape(face_grid.shape[0], -1) # 2D array with num_images rows and X*Y*3 columns (since it goes directly to FC)
    labels = labels.astype('float32') # 2D array with num_images rows and 2 columns(x, y distance from camera)

    d = [face, left_eye, right_eye, face_grid, labels]
    return d


def normalize(data, mode):
    shape = data.shape
    d = data.reshape(data.shape[0], -1) # 2D array with num_images rows and X*Y*3 columns
    d = d.astype('float32')/255 # constrain the values in [0, 1]
    d = d - np.mean(d, axis=0) # center the values around zeros
    d = d.reshape(shape) # 4D array (initial size of the array)
    if mode == 'gray':
        d = np.asarray([rgb2gray(d[i]) for i in range(d.shape[0])])
        d = d[..., None]
    return d




if __name__ == '__main__':
    readData()
