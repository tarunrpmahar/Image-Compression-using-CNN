import pickle as cPickle
from parameters import CNN_Parameter, Or_Parameter
import tensorflow as tf

good_par = Or_Parameter(verbose=False)
cnn_param = CNN_Parameter(verbose=False)


class ConvNet():

    def p(self, t):
        print(t.name, t.get_shape())

    def ld_vgg_wts(self):
        with open(good_par.vgg_weights, "rb") as f:
            self.pretrained_weights = cPickle.load(f, fix_imports=True, errors="strict")

    def cnn_build(self, image):

        image = self.img_cnvrsn_scaling(image)

        conv1_1 = self.conv_dep(image, "conv1_1", nonlinearity=tf.nn.relu)
        conv1_2 = self.conv_dep(conv1_1, "conv1_2", nonlinearity=tf.nn.relu)
        pool1 = tf.nn.max_pool2d(conv1_2, ksize=cnn_param.pool_window,strides=cnn_param.pool_stride, padding='SAME', name='pool1')

        conv2_1 = self.conv_dep(pool1, "conv2_1", nonlinearity=tf.nn.relu)
        conv2_2 = self.conv_dep(conv2_1, "conv2_2", nonlinearity=tf.nn.relu)
        pool2 = tf.nn.max_pool2d(conv2_2, ksize=cnn_param.pool_window,strides=cnn_param.pool_stride, padding='SAME', name='pool2')

        conv3_1 = self.conv_dep(pool2, "conv3_1", nonlinearity=tf.nn.relu)
        conv3_2 = self.conv_dep(conv3_1, "conv3_2", nonlinearity=tf.nn.relu)
        conv3_3 = self.conv_dep(conv3_2, "conv3_3", nonlinearity=tf.nn.relu)
        pool3 = tf.nn.max_pool2d(conv3_3, ksize=cnn_param.pool_window,strides=cnn_param.pool_stride, padding='SAME', name='pool3')

        conv4_1 = self.conv_dep(pool3, "conv4_1", nonlinearity=tf.nn.relu)
        conv4_2 = self.conv_dep(conv4_1, "conv4_2", nonlinearity=tf.nn.relu)
        conv4_3 = self.conv_dep(conv4_2, "conv4_3", nonlinearity=tf.nn.relu)
        pool4 = tf.nn.max_pool2d(conv4_3, ksize=cnn_param.pool_window,strides=cnn_param.pool_stride, padding='SAME', name='pool4')

        conv5_1 = self.conv_dep(pool4, "conv5_1", nonlinearity=tf.nn.relu)
        conv5_2 = self.conv_dep(conv5_1, "conv5_2", nonlinearity=tf.nn.relu)
        conv5_3 = self.conv_dep(conv5_2, "conv5_3", nonlinearity=tf.nn.relu)

        conv_depth_1 = self.conv_dep(conv5_3, "conv6_1")
        conv_depth = self.conv_dep(conv_depth_1, "depth")
        last_cnn = self.conv_dep(conv_depth, "conv6")
        space = tf.reduce_mean(last_cnn, [1, 2])

        with tf.compat.v1.variable_scope("GAP"):
            space_w = tf.compat.v1.get_variable("W", shape=cnn_param.layer_shapes['GAP/W'],
                                                initializer=tf.random_normal_initializer(stddev=good_par.std_dev))

        P_cls = tf.matmul(space, space_w)

        return last_cnn, space, P_cls

    def get_binary(self, tf_cls, last_cnn):
        with tf.compat.v1.variable_scope("GAP", reuse=True):
            class_w = tf.gather(tf.transpose(tf.compat.v1.get_variable("W")), tf_cls)
            class_w = tf.reshape(class_w, [-1, cnn_param.last_features, 1])
        last_cnn1 = tf.compat.v1.image.resize_bilinear(last_cnn, [good_par.image_h, good_par.image_w])
        last_cnn1 = tf.reshape(last_cnn1, [-1, good_par.image_h * good_par.image_w, cnn_param.last_features])
        binary_map = tf.reshape(tf.matmul(last_cnn1, class_w), [-1, good_par.image_h, good_par.image_w])
        return binary_map

    def get_vgg_wts(self, layer_name, bias=False):
        layer = self.pretrained_weights[layer_name]
        if bias: return layer[1]
        return layer[0].transpose((2, 3, 1, 0))

    def conv_dep(self, input_, name, nonlinearity=None):
        with tf.compat.v1.variable_scope(name) as scope:

            W_shape = cnn_param.layer_shapes[name + '/W']
            b_shape = cnn_param.layer_shapes[name + '/b']

            if good_par.fine_tuning and name not in ['conv6', 'conv6_1', 'depth']:
                W = self.get_vgg_wts(name)
                b = self.get_vgg_wts(name, bias=True)
                W_initializer = tf.constant_initializer(W)
                b_initializer = tf.constant_initializer(b)
            else:
                W_initializer = tf.truncated_normal_initializer(stddev=good_par.std_dev)
                b_initializer = tf.constant_initializer(0.0)

            conv_wts = tf.compat.v1.get_variable("W", shape=W_shape, initializer=W_initializer)
            conv_bias = tf.compat.v1.get_variable("b", shape=b_shape, initializer=b_initializer)

            if name == 'depth':
                conv = tf.compat.v1.nn.depthwise_conv2d_native(input_, conv_wts, [1, 1, 1, 1], padding='SAME')
            else:
                conv = tf.nn.conv2d(input_, conv_wts, [1, 1, 1, 1], padding='SAME')

            bias = tf.nn.bias_add(conv, conv_bias)
            bias = tf.nn.dropout(bias, 0.7)
            if nonlinearity is None:
                return bias
            return nonlinearity(bias, name=name)

    def img_cnvrsn_scaling(self, image):
        image = image*255.
        r, g, b = tf.split(image, 3, 3)
        VGG_MEAN = [103.939, 116.779, 123.68]
        return tf.concat([b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]], 3)
