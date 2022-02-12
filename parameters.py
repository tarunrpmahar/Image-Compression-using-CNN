from pprint import pprint


class CNN_Parameter():
    def __init__(self, verbose):
        self.pool_window = [1, 2, 2, 1]
        self.pool_stride = [1, 2, 2, 1]
        self.last_features = 1024
        self.conv_filters = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        self.depth_filters = [32]
        self.layer_shapes = self.get_lyr_form()

        if verbose:
            pprint(self.__dict__)

    def get_lyr_form(self):
        form = {}
        good_par = Or_Parameter(verbose=False)
        last = self.last_features
        filter = self.conv_filters
        depth = self.depth_filters[-1]

        form['conv1_1/W'] = (good_par.filter_h, good_par.filter_w, good_par.image_c, filter[0])
        form['conv1_1/b'] = (filter[0],)
        form['conv1_2/W'] = (good_par.filter_h, good_par.filter_w, filter[0], filter[1])
        form['conv1_2/b'] = (filter[1],)
        form['conv2_1/W'] = (good_par.filter_h, good_par.filter_w, filter[1], filter[2])
        form['conv2_1/b'] = (filter[2],)
        form['conv2_2/W'] = (good_par.filter_h, good_par.filter_w, filter[2], filter[3])
        form['conv2_2/b'] = (filter[3],)
        form['conv3_1/W'] = (good_par.filter_h, good_par.filter_w, filter[3], filter[4])
        form['conv3_1/b'] = (filter[4],)
        form['conv3_2/W'] = (good_par.filter_h, good_par.filter_w, filter[4], filter[5])
        form['conv3_2/b'] = (filter[5],)
        form['conv3_3/W'] = (good_par.filter_h, good_par.filter_w, filter[5], filter[6])
        form['conv3_3/b'] = (filter[6],)
        form['conv4_1/W'] = (good_par.filter_h, good_par.filter_w, filter[6], filter[7])
        form['conv4_1/b'] = (filter[7],)
        form['conv4_2/W'] = (good_par.filter_h, good_par.filter_w, filter[7], filter[8])
        form['conv4_2/b'] = (filter[8],)
        form['conv4_3/W'] = (good_par.filter_h, good_par.filter_w, filter[8], filter[9])
        form['conv4_3/b'] = (filter[9],)
        form['conv5_1/W'] = (good_par.filter_h, good_par.filter_w, filter[9], filter[10])
        form['conv5_1/b'] = (filter[10],)
        form['conv5_2/W'] = (good_par.filter_h, good_par.filter_w, filter[10], filter[11])
        form['conv5_2/b'] = (filter[11],)
        form['conv5_3/W'] = (good_par.filter_h, good_par.filter_w, filter[11], filter[12])
        form['conv5_3/b'] = (filter[12],)
        form['conv6_1/W'] = (good_par.filter_h, good_par.filter_w, filter[12], depth)
        form['conv6_1/b'] = (depth,)
        form['depth/W'] = (good_par.filter_h, good_par.filter_w, depth, depth)
        form['depth/b'] = (last,)
        form['conv6/W'] = (good_par.filter_h, good_par.filter_w, last, last)
        form['conv6/b'] = (last,)
        form['GAP/W'] = (last, good_par.n_labels)
        return form

class Training_Parameter():
    def __init__(self, verbose):
        self.model_path = './models/'
        self.num_epochs = 5
        self.learning_rate = 0.002
        self.weight_decay_rate = 0.0005
        self.momentum = 0.9
        self.batch_size = 16
        self.max_iteration = 200000
        self.test_every_iteration = 200
        self.data_train_path = './dataset/train.pickle'
        self.data_test_path = './dataset/test.pickle'
        self.images = './dataset/256_ObjectCategories'
        self.resume_training = False
        self.on_resume_fix_lr = False
        self.change_lr_env = False
        self.optimizer = 'Adam'

        if verbose:
            pprint(self.__dict__)


class Or_Parameter():
    def __init__(self, verbose):
        self.empty = True
        self.vgg_weights = 'dataset/cal_wts.pickle'
        self.model_path = 'models/wt_Model'
        self.n_labels = 257
        self.top_k = 5
        self.std_dev = 0.2
        self.fine_tuning = False
        self.image_h = 224
        self.image_w = 224
        self.image_c = 3
        self.cnn_struct = 'vgg'
        self.filter_h = 3
        self.filter_w = 3

        if verbose:
            pprint(self.__dict__)



