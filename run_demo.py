# coding:utf-8

import tensorflow as tf
from model.mrcnn import MRCNN
from model.utils import DataLoader
from model.config import Config

config = Config()

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

    model = MRCNN(
        sequence_length=dataloader.max_length,
        num_classes=num_classes,
        position_size=2 * dataloader.position_max + 2,
        position_embedding_size=config.position_embedding_size,
        word_embedding_size=dataloader.embeddings_dim,
        # word_embeddings1=dataloader.word_embeddings1,
        # word_embeddings2=dataloader.word_embeddings2,
        filter_size=config.filter_size,
        num_filters=config.num_filters,
        l2_reg_lambda=config.l2_reg_lambda,
        gamma=config.gamma,
        m_plus=config.m_plus,
        m_minus=config.m_minus,
        artificial_class_index=dataloader.artificial_class_index,
    )


    def test_step(data):
        x1_batch, x2_batch, p1_batch, p2_batch, y_batch, y_plus, c_minus = data
        test_feed_dict = {
            model.input_x1: x1_batch,
            model.input_x2: x2_batch,
            model.input_p1: p1_batch,
            model.input_p2: p2_batch,
            model.Embedding_W1: word_embeddings1,
            model.Embedding_W2: word_embeddings2,
            model.input_y: y_batch,
        }
        run_op = [model.accuracy, model.scores, model.predictions, model.dev_summary_op]
        acc, scores, predictions, summary = sess.run(run_op, feed_dict=test_feed_dict)
        return acc, scores, predictions, summary


    with tf.Session() as sess:
        saver = tf.train.Saver()  # mo saver
        sess.run(tf.global_variables_initializer())

        print('loading model...')
        saver.restore(sess, config.checkpoint_dir)
        max_dev_acc, scores_max, predictions_max = 0, None, None
        pre = 0
        scores, predictions = [], []
        for test_batch in test_data_iter:
            dev_acc, _score, _prediction, _summary = test_step(test_batch)
            pre += dev_acc
            scores += list(_score)
            predictions += list(_prediction)
        f1 = dataloader.eval(predictions)
