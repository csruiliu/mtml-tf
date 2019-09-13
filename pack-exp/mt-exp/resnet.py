import tensorflow as tf

channel_num = 3

class resnet(object):
    def __init__(self, net_name, model_layer, input_h, input_w, batch_size, num_classes, opt):
        self.net_name = net_name
        self.model_layer_num = model_layer
        self.img_h = input_h
        self.img_w = input_w
        self.input_size = input_h * input_w * channel_num
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.optimzier = opt
        self.model_size = 0
        self.cost = 0


    def conv_block(self, X_input, kernel_size, in_filter, out_filters, stage, block, training, stride):
        block_name = 'resnet' + stage + block
        f1, f2, f3 = out_filters
        layer_size = 0
        with tf.variable_scope(block_name):
            x_shortcut = X_input

            W_conv1 = self.weight_variable([1, 1, in_filter, f1])
            
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1, stride, stride, 1], padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)
            layer_size += (1 * 1 * in_filter + 1) * f1

            W_conv2 = self.weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1,1,1,1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)
            layer_size += (kernel_size * kernel_size * f1 + 1) * f2

            W_conv3 = self.weight_variable([1,1,f2,f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            layer_size += (1 * 1 * f2 + 1) * f3

            W_shortcut = self.weight_variable([1, 1, in_filter, f3])
            x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')
            layer_size += (1 * 1 * in_filter + 1) * f3

            add = tf.add(x_shortcut, X)
            add_result = tf.nn.relu(add)

        return add_result, layer_size

    def identity_block(self, X_input, kernel_size, in_filter, out_filters, stage, block, training):
        block_name = 'resnet_' + stage + '_' + block
        f1, f2, f3 = out_filters
        layer_size = 0
        with tf.variable_scope(block_name):
            X_shortcut = X_input

            W_conv1 = self.weight_variable([1, 1, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)
            layer_size += (1 * 1 * in_filter + 1) * f1

            W_conv2 = self.weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)
            layer_size += (kernel_size * kernel_size * f1 + 1) * f2

            W_conv3 = self.weight_variable([1, 1, f2, f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            layer_size += (1 * 1 * f2 + 1) * f3

            add = tf.add(X, X_shortcut)
            add_result = tf.nn.relu(add)

        return add_result, layer_size


    def build(self, input, training=True, keep_prob=0.5):
        #assert(x.shape == (x.shape[0],70,70,3))
        with tf.variable_scope(self.net_name + '_instance'):
            input_padding = input[0:self.batch_size,:,:,:]
            x = tf.pad(input_padding, tf.constant([[0, 0], [3, 3, ], [3, 3], [0, 0]]), "CONSTANT")
            #training = tf.placeholder(tf.bool, name='training')
            w_conv1 = self.weight_variable([7, 7, 3, 64])
            x = tf.nn.conv2d(x, w_conv1, strides=[1, 2, 2, 1], padding='VALID')
            x = tf.layers.batch_normalization(x, axis=3, training=training)
            x = tf.nn.relu(x)
            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            self.model_size += (7 * 7 * 3 + 1) * 64

            #stage 64
            x, x_size = self.conv_block(x, 3, 64, [64, 64, 256], stage='stage64', block='conv_block', training=training, stride=1)
            self.model_size += x_size
            x, x_size = self.identity_block(x, 3, 256, [64, 64, 256], stage='stage64', block='identity_block1', training=training)
            self.model_size += x_size
            x, x_size = self.identity_block(x, 3, 256, [64, 64, 256], stage='stage64', block='identity_block2', training=training)
            self.model_size += x_size

            #stage 128
            x, x_size = self.conv_block(x, 3, 256, [128, 128, 512], stage='stage128', block='conv_block', training=training, stride=2)
            self.model_size += x_size
            x, x_size = self.identity_block(x, 3, 512, [128, 128, 512], stage='stage128', block='identity_block1', training=training)
            self.model_size += x_size
            x, x_size = self.identity_block(x, 3, 512, [128, 128, 512], stage='stage128', block='identity_block2', training=training)
            self.model_size += x_size
            x, x_size = self.identity_block(x, 3, 512, [128, 128, 512], stage='stage128', block='identity_block3', training=training)
            self.model_size += x_size

            #stage 256
            x, x_size = self.conv_block(x, 3, 512, [256, 256, 1024], stage='stage256', block='conv_block', training=training, stride=2)
            self.model_size += x_size
            x, x_size = self.identity_block(x, 3, 1024, [256, 256, 1024], stage='stage256', block='identity_block1', training=training)
            self.model_size += x_size
            x, x_size = self.identity_block(x, 3, 1024, [256, 256, 1024], stage='stage256', block='identity_block2', training=training)
            self.model_size += x_size
            x, x_size = self.identity_block(x, 3, 1024, [256, 256, 1024], stage='stage256', block='identity_block3', training=training)
            self.model_size += x_size
            x, x_size = self.identity_block(x, 3, 1024, [256, 256, 1024], stage='stage256', block='identity_block4', training=training)
            self.model_size += x_size
            x, x_size = self.identity_block(x, 3, 1024, [256, 256, 1024], stage='stage256', block='identity_block5', training=training)
            self.model_size += x_size

            #stage 512
            x, x_size = self.conv_block(x, 3, 1024, [512, 512, 2048], stage='stage512', block='conv_block', training=training, stride=2)
            self.model_size += x_size
            x, x_size = self.identity_block(x, 3, 2048, [512, 512, 2048], stage='stage512', block='identity_block1', training=training)
            self.model_size += x_size
            x, x_size = self.identity_block(x, 3, 2048, [512, 512, 2048], stage='stage512', block='identity_block2', training=training)
            self.model_size += x_size

            avg_pool_size = int(self.img_h // 32)

            x = tf.nn.avg_pool(x, [1, avg_pool_size, avg_pool_size, 1], strides=[1,1,1,1], padding='VALID')

            flatten = tf.layers.flatten(x)
            x = tf.layers.dense(flatten, units=50, activation=tf.nn.relu)
            self.model_size += (int(flatten.shape[1]) + 1) * 50


            with tf.name_scope('dropout'):
                #keep_prob = tf.placeholder(tf.float32)
                x = tf.nn.dropout(x, keep_prob)

            logits = tf.layers.dense(x, units=self.num_classes, activation=tf.nn.softmax)
            self.model_size += (int(x.shape[1]) + 1) * self.num_classes
        
        #logit_name = self.net_name + '_instance'

        return logits
    
    def compute_grads(self, logits, labels):
        with tf.name_scope('loss_'+self.net_name):
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
            cross_entropy_cost = tf.reduce_mean(cross_entropy)
        with tf.name_scope('optimizer_'+self.net_name):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.net_name+'_instance')
            with tf.control_dependencies(update_ops):
                train_optimizer = tf.train.AdamOptimizer(1e-4) 
                #train_grads_and_vars = train_optimizer.compute_gradients(cross_entropy_cost, tf.trainable_variables())
                train_grads_and_vars = train_optimizer.compute_gradients(cross_entropy, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=self.net_name+'_instance'))
                #train_gv = [(g,v) for g,v in train_grads_and_vars if g is not None]
                #train_gv = [(g,v) for g,v in train_grads_and_vards if v :]
                #train_gv = [v for g, v in train_grads_and_vars if g is not None]
                train_step = train_optimizer.apply_gradients(train_grads_and_vars)        
        return train_grads_and_vars,  train_step


    def train_step(self, logits, labels):
        with tf.name_scope('loss_'+self.net_name):
            labels_padding = labels[0:self.batch_size:,]
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels_padding, logits=logits)
            cross_entropy_cost = tf.reduce_mean(cross_entropy)
        self.cost = cross_entropy_cost
        with tf.name_scope('optimizer_'+self.net_name):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope=self.net_name+'_instance')
            with tf.control_dependencies(update_ops):
                if self.optimzier == "Adam":
                    print("using Adam Optimizer")
                    train_optimizer = tf.train.AdamOptimizer(1e-4)
                elif self.optimzier == "SGD":
                    print("using SGD Optimizer")
                    train_optimizer = tf.train.GradientDescentOptimizer(1e-4)
        
                grads_and_vars = train_optimizer.compute_gradients(cross_entropy, tf.trainable_variables())
                train_ops = train_optimizer.apply_gradients(grads_and_vars)
        return train_optimizer, grads_and_vars, train_ops

    def train(self, logits, labels):
        with tf.name_scope('loss_'+self.net_name):
            labels_padding = labels[0:self.batch_size:,]
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels_padding, logits=logits)
        cross_entropy_cost = tf.reduce_mean(cross_entropy)
        self.cost = cross_entropy_cost
        with tf.name_scope('optimizer_'+self.net_name):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.net_name+'_instance')
            with tf.control_dependencies(update_ops):
                if self.optimzier == "Adam":
                    print("using Adam Optimizer")
                    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_cost)
                elif self.optimzier == "SGD":
                    print("using SGD Optimizer")
                    train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy_cost)
        return train_step

    def getCost(self):
        return self.cost
    
    

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def getModelInstanceName(self):
        return (self.net_name + " with layer: " + str(self.model_layer_num)) 

    def getModelMemSize(self):
        return self.model_size * 4 // (1024**2)
