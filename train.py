# coding:utf-8
import time
import os
import logging
import tensorflow as tf
from model.mrcnn import MRCNN
from model.utils import DataLoader
from model.config import Config

config = Config()
logging.basicConfig(level=logging.INFO, filename=config.log_file, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger('training')

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=config.allow_soft_placement,
        log_device_placement=config.log_device_placement
    )
    # prepare data set
    dataloader = DataLoader()
    num_classes = dataloader.numclass

    word_embeddings1 = dataloader.word_embeddings1
    word_embeddings2 = dataloader.word_embeddings2
    # train_data = dataloader.trainset
    # test_data = dataloader.testset
    # testset_size = dataloader.testset_size
    train_data_iter, train_size, train_num_batches = dataloader.batch_iter(
        is_train=True, batch_size=config.batch_size,
        num_epochs=config.num_epochs, oversample=False, shuffle=True)
    test_data_iter, test_size, test_num_batches = dataloader.batch_iter(
        is_train=False, batch_size=20,
        num_epochs=1, oversample=False, shuffle=False)
    # num_batches = math.ceil(dataloader.trainset_size / config.batch_size)
    # run_nums = config.num_epochs * num_batches

    model = MRCNN(
        sequence_length=dataloader.max_length,
        num_classes=num_classes,
        position_size=2 * dataloader.position_max + 2,
        position_embedding_size=config.position_embedding_size,
        word_embedding_size=dataloader.embeddings_dim,
        filter_size=config.filter_size,
        num_filters=config.num_filters,
        l2_reg_lambda=config.l2_reg_lambda,
        gamma=config.gamma,
        m_plus=config.m_plus,
        m_minus=config.m_minus,
        artificial_class_index=dataloader.artificial_class_index,
    )


    def input_queue(data, batch_size=32):

        x1, x2, p1, p2, y = data
        input_queue = tf.train.slice_input_producer([x1, x2, p1, p2, y], shuffle=True)
        x1_queue, x2_queue, p1_queue, p2_queue, y_queue = tf.train.batch(input_queue, batch_size=batch_size,
                                                                         capacity=64)
        return [x1_queue, x2_queue, p1_queue, p2_queue, y_queue]


    def train_step(data, learning_rate):
        x1_batch, x2_batch, p1_batch, p2_batch, y_batch, y_plus, c_minus = data
        run_op = [model.train_op, model.loss, model.accuracy, model.train_summary_op]
        feed_dict = {
            model.input_x1: x1_batch,
            model.input_x2: x2_batch,
            model.input_p1: p1_batch,
            model.input_p2: p2_batch,
            model.Embedding_W1: word_embeddings1,
            model.Embedding_W2: word_embeddings2,
            model.input_y: y_batch,
            model.input_y_plus: y_plus,
            model.input_c_minus: c_minus,
            model.learning_rate: learning_rate
        }
        _, cost, acc, merge = sess.run(run_op, feed_dict=feed_dict)
        return cost, acc, merge


    with tf.Session(config=session_conf) as sess:
        timestamp = time.strftime("%Y-%m-%d_%a_%H_%M_%S", time.localtime())
        out_dir = os.path.join(os.path.curdir, 'runs', timestamp)
        # train summaries
        train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        # dev summaries
        dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        saver = tf.train.Saver()  # mo saver
        checkpoint_dir = config.checkpoint_dir

        sess.run(tf.global_variables_initializer())

        epoch = 1
        max_dev_acc, scores_max, predictions_max = 0, None, None
        for step, train_batch in enumerate(train_data_iter, 1):
            timestring = time.strftime('%c').replace(':', '_')

            cost, acc, merge = train_step(train_batch, config.learning_rate)
            if step % config.show_step == 0:
                print('time {}, epoch {}, step {}, cost {}, acc {}'.format(timestring, epoch, step, cost, acc))
                logger.info('epoch {}, step {}, cost {:.3f}, acc {:.4f}'.format(epoch, step, cost, acc))
                train_summary_writer.add_summary(merge, step)
                train_summary_writer.flush()

            if step % train_num_batches == 0:
                epoch += 1
                # update lr
                config.learning_rate = config.learning_rate / float(epoch)
            if step % config.checkpoint_every == 0:
                save_model = saver.save(sess, checkpoint_dir)
        save_model = saver.save(sess, checkpoint_dir)
