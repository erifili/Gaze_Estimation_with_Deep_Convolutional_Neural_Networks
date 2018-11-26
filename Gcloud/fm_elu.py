import numpy as np
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter 
from tqdm import tqdm
from tensorflow.python.client import device_lib
import os
import timeit

# due to plot problems
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

        self.eye_size = 2 * 2 * 2 * self.conv4_out
        self.face_size = 2 * 2 * self.conv4_out

        # Fully-connected layers:
        self.fc_e1_size = 128
        self.fc_f1_size = 128
        self.fc_f2_size = 64
        self.fc_fg1_size = 256
        self.fc_fg2_size = 128
        self.fc1_size = 128
        self.fc2_size = 2
        
def readData(filename, flag):
    data = np.load(filename)
    if flag == 'Training':
        train_face = data['train_face']
        train_left_eye = data['train_eye_left']
        train_right_eye = data['train_eye_right']
        train_face_grid = data['train_face_mask']
        train_labels = data['train_y']
        val_face = data['val_face']
        val_left_eye = data['val_eye_left']
        val_right_eye = data['val_eye_right']
        val_face_grid = data['val_face_mask']
        val_labels = data['val_y']

        train_data = [train_face, train_left_eye, train_right_eye, train_face_grid, train_labels]
        val_data = [val_face, val_left_eye, val_right_eye, val_face_grid, val_labels]

        return train_data, val_data
    
    elif flag == 'Testing':
        test_face = data['face'][:15000]
        test_left_eye = data['leftEye'][:15000]
        test_right_eye = data['rightEye'][:15000]
        test_face_grid = data['face_grid'][:15000]
        test_labels = data['Labels'][:15000]
        
        test_data = [test_face, test_left_eye, test_right_eye, test_face_grid, test_labels]
    
        return test_data

def readChunks(file, data_size, batch):
    for i in range(0, data_size, batch):
        face = file[0][i:i+batch]
        left_eye = file[1][i:i+batch]
        right_eye = file[2][i:i+batch]
        face_grid = file[3][i:i+batch]
        labels = file[4][i:i+batch]

        data = [face, left_eye, right_eye, face_grid, labels]
        
        yield data
    
def prepare_data(data):
    face, left_eye, right_eye, face_grid, labels = data
    face = normalize(face)
    left_eye = normalize(left_eye)
    right_eye = normalize(right_eye)
    face_grid = face_grid.reshape(face_grid.shape[0], -1).astype('float32') # 2D array with num_images rows and X*Y*3 columns (since it goes directly to FC)
    labels = labels.astype('float32') # 2D array with num_images rows and 2 columns(x, y distance from camera)

    d = [face, left_eye, right_eye, face_grid, labels]
    return d

def normalize(data):
    shape = data.shape
    d = data.reshape(data.shape[0], -1) # 2D array with num_images rows and X*Y*3 columns
    d = d.astype('float32')/255 # constrain the values in [0, 1]
    d = d - np.mean(d, axis=0) # center the values around zeros
    d = d.reshape(shape) # 4D array (initial size of the array)
    
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
    return tf.nn.elu(conv2d(input_x, W, stride) + b)

def fully_conn_layer(input_layer, W, b):
    return tf.matmul(input_layer, W) + b

def plot_(train_loss_ls, train_error_ls, val_loss_ls, val_error_ls, save_file):
    # loss
    plt.figure()
    plt.subplot(211)
    plt.plot(train_loss_ls, label = 'Train loss')
    plt.plot(val_loss_ls, label = 'Val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Loss evolution")
    plt.legend()
    plt.grid(True)

    # error
    plt.subplot(212)
    plt.plot(train_error_ls, label = 'Train error')
    plt.plot(val_error_ls, label = 'Val error')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title("Error evolution")
    plt.legend()
    plt.grid(True)
    
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.subplots_adjust(top=0.92, bottom=0.1, left=0.10, right=0.95, hspace=0.6,wspace=0.35)

    if not os.path.exists('Figures'):
        os.makedirs('Figures')
        plt.savefig('Figures/%s' % save_file)
    else:
        plt.savefig('Figures/%s' % save_file)

def shuffle_data(data):
    idx = np.arange(data[0].shape[0])
    np.random.shuffle(idx)
    for i in range(len(data)):
        data[i] = data[i][idx]
    return data   


'''=================================================================================== Model =================================================================================== '''


def model(train_data, val_data, net_params, epochs, eta, train_data_size, val_data_size, batch_size, min_diff, patience, print_per_epoch, save_model_dir_fname):
    
    # # create the placeholders (Graph input)
    left_e = tf.placeholder(tf.float32, [None, net_params.img_size, net_params.img_size, net_params.channels], name='left_eye')
    right_e = tf.placeholder(tf.float32, [None, net_params.img_size, net_params.img_size, net_params.channels], name='right_eye')
    face = tf.placeholder(tf.float32, [None, net_params.img_size, net_params.img_size, net_params.channels], name='face')
    face_grid = tf.placeholder(tf.float32, [None, net_params.filter_size * net_params.filter_size], name='face_grid')
    y = tf.placeholder(tf.float32, [None, net_params.fc2_size], name='position')

    # initialize the weights and the biases.
    W = {
            'conv1_eye': init_weights([net_params.conv1_size, net_params.conv1_size, net_params.channels, net_params.conv1_out], name='convo1_eye_w'),
            'conv2_eye': init_weights([net_params.conv2_size, net_params.conv2_size, net_params.conv1_out, net_params.conv2_out], name='convo2_eye_w'),
            'conv3_eye': init_weights([net_params.conv3_size, net_params.conv3_size, net_params.conv2_out, net_params.conv3_out], name='convo3_eye_w'),
            'conv4_eye': init_weights([net_params.conv4_size, net_params.conv4_size, net_params.conv3_out, net_params.conv4_out], name='convo4_eye_w'),
            'conv1_face': init_weights([net_params.conv1_size, net_params.conv1_size, net_params.channels, net_params.conv1_out], name='convo1_face_w'),
            'conv2_face': init_weights([net_params.conv2_size, net_params.conv2_size, net_params.conv1_out, net_params.conv2_out], name='convo2_face_w'),
            'conv3_face': init_weights([net_params.conv3_size, net_params.conv3_size, net_params.conv2_out, net_params.conv3_out], name='convo3_face_w'),
            'conv4_face': init_weights([net_params.conv4_size, net_params.conv4_size, net_params.conv3_out, net_params.conv4_out], name='convo4_face_w'),
            'fc_e1': init_weights([net_params.eye_size, net_params.fc_e1_size], name='fc_e1_w'),
            'fc1_f': init_weights([net_params.face_size, net_params.fc_f1_size], name='fc1_f_w'),
            'fc2_f': init_weights([net_params.fc_f1_size, net_params.fc_f2_size], name='fc2_f_w'),
            'fc1_fg': init_weights([net_params.filter_size * net_params.filter_size, net_params.fc_fg1_size], name='fc1_fg_w'),
            'fc2_fg': init_weights([net_params.fc_fg1_size, net_params.fc_fg2_size], name='fc2_fg_w'),
            'fc1': init_weights([net_params.fc_e1_size + net_params.fc_f2_size + net_params.fc_fg2_size, net_params.fc1_size], name='fc1_w'),
            'output': init_weights([net_params.fc1_size, net_params.fc2_size], name='output_w'),
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
            'fc_e1': init_bias([net_params.fc_e1_size]),
            'fc1_f': init_bias([net_params.fc_f1_size]),
            'fc2_f': init_bias([net_params.fc_f2_size]),
            'fc1_fg': init_bias([net_params.fc_fg1_size]),
            'fc2_fg': init_bias([net_params.fc_fg2_size]),
            'fc1': init_bias([net_params.fc1_size]),
            'output': init_bias([net_params.fc2_size]),
        }
    
    
    ''' Layers: '''

    ## Convolutional layers

    #-------> left_eye path 
        
    convo1_left_eye = convolution_layer(left_e, W['conv1_eye'], b['conv1_eye'], 1)
    pooling1_left_eye = max_pooling(convo1_left_eye, net_params.pool1_size, net_params.pool1_stride)

    convo2_left_eye = convolution_layer(pooling1_left_eye, W['conv2_eye'], b['conv2_eye'], 1)
    pooling2_left_eye = max_pooling(convo2_left_eye, net_params.pool2_size, net_params.pool2_stride)

    convo3_left_eye = convolution_layer(pooling2_left_eye, W['conv3_eye'], b['conv3_eye'], 1)
    pooling3_left_eye = max_pooling(convo3_left_eye, net_params.pool3_size, net_params.pool3_stride)

    convo4_left_eye = convolution_layer(pooling3_left_eye, W['conv4_eye'], b['conv4_eye'], 1)
    pooling4_left_eye = max_pooling(convo4_left_eye, net_params.pool4_size, net_params.pool4_stride)

    pooling4_left_eye_flat = tf.reshape(pooling4_left_eye, [-1, 2 * 2 * 64])

    #-------> right_eye path
    convo1_right_eye = convolution_layer(right_e, W['conv1_eye'], b['conv1_eye'], 1)
    pooling1_right_eye = max_pooling(convo1_right_eye, net_params.pool1_size, net_params.pool1_stride)

    convo2_right_eye = convolution_layer(pooling1_right_eye, W['conv2_eye'], b['conv2_eye'], 1)
    pooling2_right_eye = max_pooling(convo2_right_eye, net_params.pool2_size, net_params.pool2_stride)

    convo3_right_eye = convolution_layer(pooling2_right_eye, W['conv3_eye'], b['conv3_eye'], 1)
    pooling3_right_eye = max_pooling(convo3_right_eye, net_params.pool3_size, net_params.pool3_stride)

    convo4_right_eye = convolution_layer(pooling3_right_eye, W['conv4_eye'], b['conv4_eye'], 1)
    pooling4_right_eye = max_pooling(convo4_right_eye, net_params.pool4_size, net_params.pool4_stride)


    pooling4_right_eye_flat = tf.reshape(pooling4_right_eye, [-1, 2 * 2 * 64])


    #-------> face path
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

    #-------> eyes
    fc_e1_in = tf.concat([pooling4_left_eye_flat, pooling4_right_eye_flat], 1)
    fc_e1 = tf.nn.elu(fully_conn_layer(fc_e1_in, W['fc_e1'], b['fc_e1']))

    #-------> face
    fc1_f = tf.nn.elu(fully_conn_layer(pooling4_f_flat, W['fc1_f'], b['fc1_f']))
    fc2_f = tf.nn.elu(fully_conn_layer(fc1_f, W['fc2_f'], b['fc2_f']))

    #-------> face grid
    fc1_fg = tf.nn.elu(fully_conn_layer(face_grid, W['fc1_fg'], b['fc1_fg']))
    fc2_fg = tf.nn.elu(fully_conn_layer(fc1_fg, W['fc2_fg'], b['fc2_fg']))

    ## output
    fc = tf.concat([fc_e1, fc2_f, fc2_fg], 1)
    fc1 = tf.nn.elu(fully_conn_layer(fc, W['fc1'], b['fc1']))
    y_pred = fully_conn_layer(fc1, W['output'], b['output'])
    


    ''' Set Loss, Optimizer, Error functions '''

    loss = tf.losses.mean_squared_error(y, y_pred)
    optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(loss)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=eta).minimize(loss)
    error = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(y, y_pred), axis=1)))

    train_loss_ls, train_error_ls = [], []
    val_loss_ls, val_error_ls = [], []
    error_increment = 0
    best_loss = np.Inf
        
    """*******************************"""
    # Create the collection to restore for prediction
    tf.get_collection("output_layer")
    tf.add_to_collection("output_layer", y_pred)
    """*******************************"""
    
    # saving the model:
    saver = tf.train.Saver(max_to_keep=1)

    ## Initialize variables
    init = tf.global_variables_initializer()
    ## Session
    with tf.Session() as sess:
        sess.run(init)
        
        for ep in tqdm(range(1, epochs + 1)):
            error_increment += 1
            train_loss, train_error  = 0, 0
            val_loss, val_error = 0, 0

            # shuffling the training data    
            train_data = shuffle_data(train_data)

            n_train_batches = train_data_size / batch_size + (train_data_size % batch_size != 0)
            for batch in readChunks(train_data, train_data_size, batch_size):
                face_batch = batch[0]
                left_eye = batch[1]
                right_eye = batch[2]
                faceGrid_batch = batch[3]
                y_batch = batch[4]
                
                sess.run(optimizer, feed_dict={face: face_batch, left_e:left_eye, right_e:right_eye, face_grid: faceGrid_batch,  y: y_batch})
                                
                train_batch_loss, train_batch_error = sess.run([loss, error], feed_dict={face: face_batch, left_e:left_eye, right_e:right_eye, face_grid: faceGrid_batch,  y: y_batch})
                train_loss += train_batch_loss / n_train_batches
                train_error += train_batch_error / n_train_batches

            
            n_val_batches = val_data_size / batch_size + (val_data_size % batch_size != 0)
            for batch in readChunks(val_data, val_data_size, batch_size):
                face_batch_val = batch[0]
                left_eye_val = batch[1]
                right_eye_val = batch[2]
                faceGrid_batch_val = batch[3]
                y_batch_val = batch[4]
                val_batch_loss, val_batch_error = sess.run([loss, error], feed_dict={face: face_batch_val, left_e:left_eye_val, right_e:right_eye_val, face_grid: faceGrid_batch_val,  y: y_batch_val})
                val_loss += val_batch_loss / n_val_batches
                val_error += val_batch_error / n_val_batches

            # could not run in K80: memory errors etc, so we went with batches instead (see above)
            # val_loss, val_error = sess.run([loss, error], feed_dict={face: val_data[0], left_e: val_data[1], right_e: val_data[2], face_grid: val_data[3], y: val_data[4]})

            train_loss_ls.append(train_loss)
            train_error_ls.append(train_error)
            val_loss_ls.append(val_loss)
            val_error_ls.append(val_error)

            if val_loss - min_diff < best_loss:
                    best_loss = val_loss

                    model_dir = os.path.split(save_model_dir_fname)[0]
                    if not os.path.exists('Saved_Models'):
                        os.makedirs('Saved_Models')
                        os.makedirs('./Saved_Models/%s' % model_dir)
                        
                        saver.save(sess, './Saved_Models/%s' % save_model_dir_fname) #, global_step = ep)      
                    else:
                        if not os.path.exists('./Saved_Models/%s' % model_dir):
                            os.makedirs('./Saved_Models/%s' % model_dir)
                            saver.save(sess, './Saved_Models/%s' % save_model_dir_fname) #, global_step = ep)      
                        else:
                            saver.save(sess, './Saved_Models/%s' % save_model_dir_fname) #, global_step = ep)
                    
                    error_increment = 0

            if ep % print_per_epoch == 0:
                # control the prints to std output
                print('train loss: %.5f, val loss: %.5f, train error: %.5f, val error: %.5f' % (train_loss, val_loss, train_error, val_error))
                      
            if error_increment >= patience:
                print('Early stopping occured!')            
                if not os.path.exists('./Records'):
                        os.makedirs('Records')
                        with open('./Records/records.txt', 'a') as f:
                            f.write('Epochs: %d, eta: %f, batch_size: %d  \n' % (ep, eta, batch_size))
                            f.write('\t train loss: %.5f, val loss: %.5f, train error: %.5f, val error: %.5f \n' % (train_loss, val_loss, train_error, val_error))
                            
                
                else:
                    with open('./Records/records.txt', 'a') as f:
                            f.write('Epochs: %d, eta: %f, batch_size: %d  \n' % (ep, eta, batch_size))
                            f.write('\t train loss: %.5f, val loss: %.5f, train error: %.5f, val error: %.5f \n' % (train_loss, val_loss, train_error, val_error))
                
                return train_loss_ls, train_error_ls, val_loss_ls, val_error_ls
        

        if not os.path.exists('./Records'):
            os.makedirs('Records')
            with open('./Records/records.txt', 'a') as f:
                f.write('Epochs: %d, eta: %f, batch_size: %d  \n' % (ep, eta, batch_size))
                f.write('\t train loss: %.5f, val loss: %.5f, train error: %.5f, val error: %.5f \n' % (train_loss, val_loss, train_error, val_error))
                
        
        else:
            with open('./Records/records.txt', 'a') as f:
                f.write('Epochs: %d, eta: %f, batch_size: %d  \n' % (ep, eta, batch_size))
                f.write('\t train loss: %.5f, val loss: %.5f, train error: %.5f, val error: %.5f \n' % (train_loss, val_loss, train_error, val_error))

        return train_loss_ls, train_error_ls, val_loss_ls, val_error_ls



'''=================================================================================== Main =================================================================================== '''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tr', action='store_true', help='Training model')
    parser.add_argument('--te', action='store_true', help='Testing model')
    parser.add_argument('-sm', '--save_model_dir_fname', type=str, help='Dir to save the model and filename')
    parser.add_argument('-lm', '--load_model', type=str, help='Dir to restore the model')
    parser.add_argument('-pn', '--plot_fname', default = 'Loss.png', type=str, help='Filename to save the plot')
    args = parser.parse_args()
    
    if args.tr:
        if not args.save_model_dir_fname:
            print("Training without saving the model..")

        # load Data
        print('Loading and Preparing data...')
        start = timeit.default_timer()
        filename = './data/eye_tracker_train_and_val.npz'
        train_data, val_data = readData(filename, flag = 'Training')
        train_data = prepare_data(train_data)
        val_data = prepare_data(val_data)
        print('Data loaded after %.1fs' % (timeit.default_timer() - start))
        
        # get model params
        net_params = NetParams()
        
        # train model
        train_loss_ls, train_error_ls, val_loss_ls, val_error_ls = model(train_data, val_data, net_params, epochs = 100,
                                                                                                        eta = 0.001,
                                                                                                        train_data_size = 48000,
                                                                                                        val_data_size = 5000, 
                                                                                                        batch_size = 128, 
                                                                                                        min_diff = 1e-4, 
                                                                                                        patience = 5, 
                                                                                                        print_per_epoch = 1,
                                                                                                        save_model_dir_fname = args.save_model_dir_fname)

    
        plot_(train_loss_ls, train_error_ls, val_loss_ls, val_error_ls, save_file = args.plot_fname)
    
    elif args.te:
        print('Loading and Preparing data...')
        start = timeit.default_timer()
        filename = './data/test.npz'
        test_data = readData(filename, flag = 'Testing')
        test_data = prepare_data(test_data)
        print('Data loaded after %.1fs' % (timeit.default_timer() - start))
        
        # start the session
        sess=tf.Session() 
        
        # Load meta graph and restore the model
        net = tf.train.import_meta_graph(args.load_model + '.meta')
        net.restore(sess, args.load_model)
        
        
        # Now, let's access and create placeholders variables and
        # create feed-dict to feed new data
        
        graph = tf.get_default_graph()
        left_e_test = graph.get_tensor_by_name('left_eye:0')
        right_e_test = graph.get_tensor_by_name('right_eye:0')
        face_test = graph.get_tensor_by_name('face:0')
        face_grid_test = graph.get_tensor_by_name('face_grid:0')
        y = graph.get_tensor_by_name('position:0')

        # get the output layer
        pred = tf.get_collection_ref("output_layer")[0]
        
        # set the error function
        # error = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(y, pred), axis=1)))
        # n_test_batches = test_data[0].shape[0] / batch_size + (test_data[0].shape[0] % batch_size != 0)

        # total_test_error = 0
        test_error_ls = []
        batch_size = 128

        
        for batch in readChunks(test_data, test_data[0].shape[0], batch_size):
            face_batch_test = batch[0]
            left_eye_test = batch[1]
            right_eye_test = batch[2]
            faceGrid_batch_test = batch[3]
            y_batch_test = batch[4]
            y_pred = sess.run(pred, feed_dict={face_test: face_batch_test, left_e_test:left_eye_test, right_e_test:right_eye_test, face_grid_test: faceGrid_batch_test,  y: y_batch_test})
            # total_test_error += test_error / n_test_batches
            error = np.mean(np.sqrt(np.sum((y_pred - y_batch_test)**2, axis=1)))
            test_error_ls.append(error)

        # Compute total error
        total_test_error = np.mean(test_error_ls)
        
        # Could not run in K80: memory errors etc, so we went with batches instead (see below)
        # feed_dict ={
        #             face_test: test_data[0], left_e_test: test_data[1], 
        #             right_e_test: test_data[2], face_grid_test: test_data[3], 
        #             y:test_data[4]
        #             }
        # total_test_error = sess.run(error, feed_dict=feed_dict)

        with open('./Records/records.txt', 'a') as f:
                f.write('\t Test error: %f \n' % total_test_error)
            
        print('Test error: %f' % total_test_error)
      
    else:
        if not args.te and not args.tr:
            raise Exception("Wrong argument... try again and choose between '--tr' or '--te'")
    

if __name__ == '__main__':
    main()
