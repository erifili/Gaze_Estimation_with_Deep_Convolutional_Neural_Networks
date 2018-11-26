import numpy as np
import tensorflow as tf
# from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from tqdm import tqdm
import timeit
from tensorflow.python.client import device_lib
import os

plt.switch_backend('agg')

# check which devices you use:
print(device_lib.list_local_devices())


'''=================================================================================== Helper functions =================================================================================== '''
class NetParams():
    
    def __init__(self):
        # Net params
        self.img_size = 64
        self.channels = 3
        self.filter_size = 25

        # left/right eyes and face:
        self.conv1_size = 11
        self.conv1_out = 96
        self.pool1_size = 2
        self.pool1_stride = 2

        self.conv2_size = 5
        self.conv2_out = 256
        self.pool2_size = 2
        self.pool2_stride = 2

        self.conv3_size = 3
        self.conv3_out = 384
        self.pool3_size = 2
        self.pool3_stride = 2

        self.conv4_size = 1
        self.conv4_out = 64
        self.pool4_size = 2
        self.pool4_stride = 2

        # Fully-connected layers:
        self.fc_e1_size = 128
        self.fc_f1_size = 128
        self.fc_f2_size = 64
        self.fc_fg1_size = 256
        self.fc_fg2_size = 128
        self.fc1_size = 128
        self.fc2_size = 2
        
        


def readChunks(file, data_size, batch, flag):
    if flag == 'Train':
        for i in range(0, data_size, batch):
            train_face = file['train_face'][i:i+batch]
            train_left_eye = file['train_eye_left'][i:i+batch]
            train_right_eye = file['train_eye_right'][i:i+batch]
            train_face_grid = file['train_face_mask'][i:i+batch]
            train_labels = file['train_y'][i:i+batch]


            train_data = [train_face, train_left_eye, train_right_eye, train_face_grid, train_labels]
            train_data = prepare_data(train_data, '')
            
            yield train_data
    
    elif flag == 'Validation':
        for i in range(0, data_size, batch):
            val_face = file['val_face'][i:i+batch]
            val_left_eye = file['val_eye_left'][i:i+batch]
            val_right_eye = file['val_eye_right'][i:i+batch]
            val_face_grid = file['val_face_mask'][i:i+batch]
            val_labels = file['val_y'][i:i+batch]


            val_data = [val_face, val_left_eye, val_right_eye, val_face_grid, val_labels]
            val_data = prepare_data(val_data, '')
            
            yield val_data

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

def init_weights(shape, name):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def init_bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W, stride):
    # x ---> [batch, height, width, Channels]
    # W ---> [filter height, filter width, Channels IN, Channels OUT ]
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

def max_pooling(x, pool_size, stride):
    # x ---> [batch, height, width, Channels]
    return tf.nn.max_pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, stride, stride, 1], padding='VALID')

def convolution_layer(input_x, W, b, stride):
    return tf.nn.relu(conv2d(input_x, W, stride) + b)

def fully_conn_layer(input_layer, size, name):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size], name)
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b

def plot_loss(train_loss, train_err, test_err, start=0, per=1):
    assert len(train_err) == len(test_err)
    idx = np.arange(start, len(train_loss), per)
    fig, ax1 = plt.subplots()
    lns1 = ax1.plot(idx, train_loss[idx], 'b-', alpha=1.0, label='train loss')
    ax1.set_xlabel('epochs')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('loss', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    lns2 = ax2.plot(idx, train_err[idx], 'r-', alpha=1.0, label='train error')
    lns3 = ax2.plot(idx, test_err[idx], 'g-', alpha=1.0, label='test error')
    ax2.set_ylabel('error', color='r')
    ax2.tick_params('y', colors='r')

    # added these three lines
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)

    fig.tight_layout()
    if not os.path.exists('Figures'):
        os.makedirs('Figures')
        plt.savefig('./Figures/loss.png')
    else:
        plt.savefig('./Figures/loss.png')
    # plt.show()

'''=================================================================================== Model =================================================================================== '''


def model(data, W, b, net_params, epochs,  train_data_size, val_data_size, batch_size, min_diff, patience, print_per_epoch):

    # # create the placeholders (Graph input)
    left_e = tf.placeholder(tf.float32, [None, net_params.img_size, net_params.img_size, net_params.channels], name='left_eye')
    right_e = tf.placeholder(tf.float32, [None, net_params.img_size, net_params.img_size, net_params.channels], name='right_eye')
    face = tf.placeholder(tf.float32, [None, net_params.img_size, net_params.img_size, net_params.channels], name='face')
    face_grid = tf.placeholder(tf.float32, [None, net_params.filter_size * net_params.filter_size], name='face_grid')
    y = tf.placeholder(tf.float32, [None, net_params.fc2_size], name='position')

    ## Layers:

    ## Convolutional layers

    ###### left_eye path 
        
    convo1_left_eye = convolution_layer(left_e, W['conv1_eye'], b['conv1_eye'], 1)
    pooling1_left_eye = max_pooling(convo1_left_eye, net_params.pool1_size, net_params.pool1_stride)

    convo2_left_eye = convolution_layer(pooling1_left_eye, W['conv2_eye'], b['conv2_eye'], 1)
    pooling2_left_eye = max_pooling(convo2_left_eye, net_params.pool2_size, net_params.pool2_stride)

    convo3_left_eye = convolution_layer(pooling2_left_eye, W['conv3_eye'], b['conv3_eye'], 1)
    pooling3_left_eye = max_pooling(convo3_left_eye, net_params.pool3_size, net_params.pool3_stride)

    convo4_left_eye = convolution_layer(pooling3_left_eye, W['conv4_eye'], b['conv4_eye'], 1)
    pooling4_left_eye = max_pooling(convo4_left_eye, net_params.pool4_size, net_params.pool4_stride)

    pooling4_left_eye_flat = tf.reshape(pooling4_left_eye, [-1, 2 * 2 * 64])

    ###### right_eye path
    convo1_right_eye = convolution_layer(right_e, W['conv1_eye'], b['conv1_eye'], 1)
    pooling1_right_eye = max_pooling(convo1_right_eye, net_params.pool1_size, net_params.pool1_stride)

    convo2_right_eye = convolution_layer(pooling1_right_eye, W['conv2_eye'], b['conv2_eye'], 1)
    pooling2_right_eye = max_pooling(convo2_right_eye, net_params.pool2_size, net_params.pool2_stride)

    convo3_right_eye = convolution_layer(pooling2_right_eye, W['conv3_eye'], b['conv3_eye'], 1)
    pooling3_right_eye = max_pooling(convo3_right_eye, net_params.pool3_size, net_params.pool3_stride)

    convo4_right_eye = convolution_layer(pooling3_right_eye, W['conv4_eye'], b['conv4_eye'], 1)
    pooling4_right_eye = max_pooling(convo4_right_eye, net_params.pool4_size, net_params.pool4_stride)


    pooling4_right_eye_flat = tf.reshape(pooling4_right_eye, [-1, 2 * 2 * 64])


    ###### face path
    convo1_f = convolution_layer(face, W['conv1_face'], b['conv1_face'], 1)
    pooling1_f = max_pooling(convo1_f, net_params.pool1_size, net_params.pool1_stride)

    convo2_f = convolution_layer(pooling1_f, W['conv2_face'], b['conv2_face'], 1)
    pooling2_f = max_pooling(convo2_f, net_params.pool2_size, net_params.pool2_stride)

    convo3_f = convolution_layer(pooling2_f, W['conv3_face'], b['conv3_face'], 1)
    pooling3_f = max_pooling(convo3_f, net_params.pool3_size, net_params.pool3_stride)

    convo4_f = convolution_layer(pooling3_f, W['conv4_face'], b['conv4_face'], 1)
    pooling4_f = max_pooling(convo4_f, net_params.pool4_size, net_params.pool4_stride)

    pooling4_f_flat = tf.reshape(pooling4_f, [-1, 2 * 2 * 64])


    ## Fully-Connected layers

    ###### eyes
    fc_e1_in = tf.concat([pooling4_left_eye_flat, pooling4_right_eye_flat], 1)
    fc_e1 = tf.nn.relu(fully_conn_layer(fc_e1_in, net_params.fc_e1_size, name = 'fc_e1'))

    ###### face
    fc1_f = tf.nn.relu(fully_conn_layer(pooling4_f_flat, net_params.fc_f1_size, name = 'fc1_f'))
    fc2_f = tf.nn.relu(fully_conn_layer(fc1_f, net_params.fc_f2_size, name = 'fc2_f'))

    ###### face grid
    fc1_fg = tf.nn.relu(fully_conn_layer(face_grid, net_params.fc_fg1_size, name = 'fc1_fg'))
    fc2_fg = tf.nn.relu(fully_conn_layer(fc1_fg, net_params.fc_fg2_size, name = 'fc2_fg'))

    ## output
    fc = tf.concat([fc_e1, fc2_f, fc2_fg], 1)
    fc1 = tf.nn.relu(fully_conn_layer(fc, net_params.fc1_size, name = 'fc1'))
    y_pred = tf.nn.relu(fully_conn_layer(fc1, net_params.fc2_size, name = 'output'))


    ## Set Loss, Optimizer, Error
    loss = tf.losses.mean_squared_error(y, y_pred)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0008).minimize(loss)
    error = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(y, y_pred), axis=1)))


    ## Initialize variables
    init = tf.global_variables_initializer()

    train_loss_ls, train_error_ls = [], []
    val_loss_ls, val_error_ls = [], []
    error_increment = 0
    best_loss = np.Inf
    n_train_batches = train_data_size / batch_size + (train_data_size % batch_size != 0)
    n_val_batches = val_data_size / batch_size + (val_data_size % batch_size != 0)
    


    # saving the model:
    saver = tf.train.Saver()
    

    with tf.Session() as sess:
        sess.run(init)
        
        for ep in tqdm(range(epochs)):
            error_increment += 1
            train_loss, train_error = 0, 0
            val_loss, val_error = 0, 0

            # think of shuffling the data here
            # ...

            # start = timeit.default_timer()
            for batch in readChunks(data, train_data_size, batch_size, flag = 'Train'):
                face_batch = batch[0]
                left_eye = batch[1]
                right_eye = batch[2]
                faceGrid_batch = batch[3]
                y_batch = batch[4]
                # print('runtime: %.1fs' % (timeit.default_timer() - start))
                sess.run(optimizer, feed_dict={face: face_batch, left_e:left_eye, right_e:right_eye, face_grid: faceGrid_batch,  y: y_batch})
                
                train_batch_loss, train_batch_error = sess.run([loss, error], feed_dict={face: face_batch, left_e:left_eye, right_e:right_eye, face_grid: faceGrid_batch,  y: y_batch})
                train_loss += train_batch_loss / n_train_batches
                train_error += train_batch_error / n_train_batches

            
            for batch in readChunks(data, val_data_size, batch_size, flag = 'Validation'):
                face_batch_val = batch[0]
                left_eye_val = batch[1]
                right_eye_val = batch[2]
                faceGrid_batch_val = batch[3]
                y_batch_val = batch[4]
                val_batch_loss, val_batch_error = sess.run([loss, error], feed_dict={face: face_batch_val, left_e:left_eye_val, right_e:right_eye_val, face_grid: faceGrid_batch_val,  y: y_batch_val})
                val_loss += val_batch_loss / n_val_batches
                val_error += val_batch_error / n_val_batches


            train_loss_ls.append(train_loss)
            train_error_ls.append(train_error)
            val_loss_ls.append(val_loss)
            val_error_ls.append(val_error)

            if val_loss - min_diff < best_loss:
                    best_loss = val_loss
                    if not os.path.exists('Saved_Models'):
                        os.makedirs('Saved_Models')
                        saver.save(sess, './Saved_Models/full_model.ckpt', global_step = ep)      
                    else:
                        saver.save(sess, './Saved_Models/full_model.ckpt', global_step = ep)
                    
                    error_increment = 0

            if ep % print_per_epoch == 0:
                print('train loss: %.5f, train error: %.5f, val loss: %.5f, val error: %.5f' % (train_loss, train_error, val_loss, val_error))
                      
            if error_increment >= patience:
                print('Early stopping occured!')
                return train_loss_ls, train_error_ls, val_loss_ls, val_error_ls
        
        return train_loss_ls, train_error_ls, val_loss_ls, val_error_ls



'''=================================================================================== Main =================================================================================== '''

def main():
    data = np.load('./data/eye_tracker_train_and_val.npz')
    net_params = NetParams()
    
    # initialize the weights and the biases for the convolutional layers
    W = {
            'conv1_eye': init_weights([net_params.conv1_size, net_params.conv1_size, net_params.channels, net_params.conv1_out], name='convo1_eye'),
            'conv2_eye': init_weights([net_params.conv2_size, net_params.conv2_size, net_params.conv1_out, net_params.conv2_out], name='convo2_eye'),
            'conv3_eye': init_weights([net_params.conv3_size, net_params.conv3_size, net_params.conv2_out, net_params.conv3_out], name='convo3_eye'),
            'conv4_eye': init_weights([net_params.conv4_size, net_params.conv4_size, net_params.conv3_out, net_params.conv4_out], name='convo4_eye'),
            'conv1_face': init_weights([net_params.conv1_size, net_params.conv1_size, net_params.channels, net_params.conv1_out], name='convo1_face'),
            'conv2_face': init_weights([net_params.conv2_size, net_params.conv2_size, net_params.conv1_out, net_params.conv2_out], name='convo2_face'),
            'conv3_face': init_weights([net_params.conv3_size, net_params.conv3_size, net_params.conv2_out, net_params.conv3_out], name='convo3_face'),
            'conv4_face': init_weights([net_params.conv4_size, net_params.conv4_size, net_params.conv3_out, net_params.conv4_out], name='convo4_face'),
        }

    b = {
            'conv1_eye': init_bias([net_params.conv1_out]),
            'conv2_eye': init_bias([net_params.conv2_out]),
            'conv3_eye': init_bias([net_params.conv3_out]),
            'conv4_eye': init_bias([net_params.conv4_out]),
            'conv1_face': init_bias([net_params.conv1_out]),
            'conv2_face': init_bias([net_params.conv2_out]),
            'conv3_face': init_bias([net_params.conv3_out]),
            'conv4_face': init_bias([net_params.conv4_out]),
            
        }



    # train model
    train_loss_ls, train_error_ls, val_loss_ls, val_error_ls = model(data, W, b, net_params, epochs = 15, 
                                                               train_data_size = 1000,
                                                               val_data_size = 300, 
                                                               batch_size = 100, 
                                                               min_diff = 1e-4, 
                                                               patience = 10, 
                                                               print_per_epoch = 1)

    
    plot_loss(np.array(train_loss_ls), np.array(train_error_ls), np.array(val_error_ls), start=0, per=1)
    plt.figure() 
    plt.plot(train_loss_ls, label = 'Train loss')
    plt.plot(val_loss_ls, label = 'Val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Loss evolution")
    plt.legend()
    plt.savefig('./Figures/myLoss.png')

if __name__ == '__main__':
    main()
