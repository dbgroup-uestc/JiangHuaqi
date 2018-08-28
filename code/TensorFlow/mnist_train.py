import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib as tc
import  mnist_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZION_RATE = 0.0001
TRANING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./path/to/model"
MODEL_NAME = "model.ckpt"

def train(mnist):
    x = tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name="x-input")
    y_  = tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NODE],name="y-input")
    regulizer = tc.layers.l2_regularizer(REGULARAZION_RATE)
    y = mnist_inference.inference(x,regulizer)
    global_step = tf.Variable(0,trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.arg_max(y_,1)))
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples / BATCH_SIZE,LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_mean,global_step)
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name="train")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRANING_STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            _,loss_value,step,ty = sess.run([train_op,loss,global_step,y],feed_dict={x:xs,y_:ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batcg is %g." % (step,loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

def main(argv=None):
        mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
        train(mnist)

if __name__ == '__main__':
        main()