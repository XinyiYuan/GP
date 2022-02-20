import argparse

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
from data_utils import *
from os.path import join
import matplotlib.pyplot as plt
from datetime import datetime

def load_data_train(add_real, add_fake, block_size):
    tmp_samples, tmp_samples_diff, tmp_labels = get_data(join(add_fake, "train/"), 1, block_size)
    train_samples, train_samples_diff, train_labels = get_data(join(add_real, "train/"), 0, block_size)

    train_samples = np.concatenate((train_samples, tmp_samples), axis=0)
    train_samples_diff = np.concatenate((train_samples_diff, tmp_samples_diff), axis=0)
    train_labels = np.concatenate((train_labels, tmp_labels), axis=0)

    # We need to copy this labels to _diff for we need to shuffle them separately
    train_labels_diff = train_labels.copy()

    """
    Shuffle the training data
    """
    np.random.seed(200)
    np.random.shuffle(train_samples)
    np.random.seed(200)
    np.random.shuffle(train_labels)

    np.random.seed(500)
    np.random.shuffle(train_samples_diff)
    np.random.seed(500)
    np.random.shuffle(train_labels_diff)

    """
    Flush the memory
    """
    tmp_samples = []
    tmp_samples_diff = []
    tmp_labels = []

    return train_samples, train_samples_diff, train_labels, train_labels_diff


def load_data_test(add_real, add_fake, block_size):
    test_samples, test_samples_diff, test_labels, test_labels_video, test_sv, test_vc = \
        get_data_for_test(join(add_real, "test/"), 0, block_size)
    tmp_samples, tmp_samples_diff, tmp_labels, tmp_labels_video, tmp_sv, tmp_vc = \
        get_data_for_test(join(add_fake + "test/"), 1, block_size)

    test_samples = np.concatenate((test_samples, tmp_samples), axis=0)
    test_samples_diff = np.concatenate((test_samples_diff, tmp_samples_diff), axis=0)
    test_labels = np.concatenate((test_labels, tmp_labels), axis=0)
    test_labels_video = np.concatenate((test_labels_video, tmp_labels_video), axis=0)
    test_sv = np.concatenate((test_sv, tmp_sv), axis=0)

    test_vc.update(tmp_vc)

    """
    Flush the memory
    """
    tmp_samples = []
    tmp_samples_diff = []
    tmp_labels = []

    return test_samples, test_samples_diff, test_labels, test_labels_video, test_sv, test_vc


def main(args):
    if_train = args.train
    if_evaluate = args.evaluate
    if_gpu = args.gpu
    
    if if_train:
        if_evaluate = True

    """
    Initialization
    """
    BLOCK_SIZE = 60
    EPOCHS1 = args.epochs1
    EPOCHS2 = args.epochs2
    BATCH_SIZE = args.batch_size
    DROPOUT_RATE1 = args.dropout_rate1
    DROPOUT_RATE2 = args.dropout_rate2
    DROPOUT_RATE3 = args.dropout_rate3
    DROPOUT_RATE4 = args.dropout_rate4
    RNN_UNIT1 = args.rnn_unit1
    RNN_UNIT2 = args.rnn_unit2

    add_real = './datasets/real/'
    add_fake = './datasets/fake/'
    local_time = datetime.now().strftime('%m-%d-%H-%M')
    
    
    if if_gpu:
        # Optional to uncomment if some bugs occur.
        # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(device=gpu, enable=True)
        device = "CPU" if len(gpus) == 0 else "GPU"
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = 'CPU'

    """
    Define the models.
    # model = g1, model_diff = g2
    """
    model = K.Sequential([
        layers.InputLayer(input_shape=(BLOCK_SIZE, 1404)), # 468*3
        layers.Dropout(DROPOUT_RATE1),
        layers.Bidirectional(layers.GRU(RNN_UNIT1)),
        layers.Dropout(DROPOUT_RATE2),
        layers.Dense(RNN_UNIT1, activation='relu'),
        layers.Dropout(DROPOUT_RATE2),
        layers.Dense(2, activation='softmax')
    ])
    model_diff = K.Sequential([
        layers.InputLayer(input_shape=(BLOCK_SIZE - 1, 1404)),
        layers.Dropout(DROPOUT_RATE3),
        layers.Bidirectional(layers.GRU(RNN_UNIT2)),
        layers.Dropout(DROPOUT_RATE4),
        layers.Dense(RNN_UNIT2, activation='relu'),
        layers.Dropout(DROPOUT_RATE4),
        layers.Dense(2, activation='softmax')
    ])

    lossFunction = K.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = K.optimizers.Adam(learning_rate=0.001)
    
    if if_train:
        add_weights = './time=' + local_time + '/'
        os.mkdir("./time=" + local_time)
        train_samples, train_samples_diff, train_labels, train_labels_diff = \
            load_data_train(add_real, add_fake, BLOCK_SIZE)
        test_samples, test_samples_diff, test_labels, test_labels_video, test_sv, test_vc = \
            load_data_test(add_real, add_fake, BLOCK_SIZE)
            
        # ----For g1----#
        callbacks = [
            K.callbacks.ModelCheckpoint(
                filepath=add_weights + 'g1.h5',
                save_best_only=True,
                monitor='val_acc',
                verbose=1)
        ]
        model.compile(optimizer=optimizer, loss=lossFunction, metrics=['acc'])
        history = model.fit(train_samples, train_labels, batch_size=BATCH_SIZE,
                  validation_data=(test_samples, test_labels), epochs=EPOCHS1,
                  shuffle=True, callbacks=callbacks)
            
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, EPOCHS1+1)
        
        plt.figure(1)
        plt.title('g1: Accuracy and Loss')
        plt.plot(epochs, acc, 'red', label='acc')
        plt.plot(epochs, val_acc, 'green', label='val_acc')
        plt.plot(epochs, loss, 'blue', label='loss')
        plt.plot(epochs, val_loss, 'yellow', label='val_loss')
        plt.ylim(0.3, 1)
        plt.legend()
        plt.savefig('./time=' + local_time + '/model.jpg')
        # ----g1 end----#


        # ----For g2----#
        callbacks_diff = [
            K.callbacks.ModelCheckpoint(
                filepath=add_weights + 'g2.h5',
                save_best_only=True,
                monitor='val_acc',
                verbose=1)
        ]
        model_diff.compile(optimizer=optimizer, loss=lossFunction, metrics=['acc'])
        history_diff = model_diff.fit(train_samples_diff, train_labels_diff, batch_size=BATCH_SIZE,
                       validation_data=(test_samples_diff, test_labels), epochs=EPOCHS2,
                       shuffle=True, callbacks=callbacks_diff)
        
        acc_diff = history_diff.history['acc']
        val_acc_diff = history_diff.history['val_acc']
        loss_diff = history_diff.history['loss']
        val_loss_diff = history_diff.history['val_loss']
        epochs_diff = range(1, EPOCHS2+1)
        
        plt.figure(2)
        plt.title('g2: Accuracy and Loss')
        plt.plot(epochs_diff, acc_diff, 'red', label='acc')
        plt.plot(epochs_diff, val_acc_diff, 'green', label='val_acc')
        plt.plot(epochs_diff, loss_diff, 'blue', label='loss')
        plt.plot(epochs_diff, val_loss_diff, 'yellow', label='val_loss')
        plt.ylim(0.3, 1)
        plt.legend()
        plt.savefig('./time=' + local_time + '/model_diff.jpg')
        # ----g2 end----#
        
        
    if if_evaluate:
        if not if_train:
            add_weights = './weights/'
        
        test_samples, test_samples_diff, test_labels, test_labels_video, test_sv, test_vc = \
            load_data_test(add_real, add_fake, BLOCK_SIZE)
        
        # ----For g1----#
        model.compile(optimizer=optimizer, loss=lossFunction, metrics=['acc'])
        model.load_weights(add_weights + 'g1.h5')
        loss_g1, acc_g1 = model.evaluate(test_samples, test_labels, batch_size=512)
        # ----g1 end----#
        
        # ----For g2----#
        model_diff.compile(optimizer=optimizer, loss=lossFunction, metrics=['acc'])
        model_diff.load_weights(add_weights + 'g2.h5')
        loss_g2, acc_g2 = model_diff.evaluate(test_samples_diff, test_labels, batch_size=512)
        # ----g2 end----#
        
        """
        Evaluate the merged prediction (sample-level and video-level)
        """
            
        # ----Sample-level----#
        prediction = model.predict(test_samples)
        prediction_diff = model_diff.predict(test_samples_diff)
        count_s = 0
        total_s = test_labels.shape[0]
        mix_predict = []
        for i in range(len(prediction)):
            mix = prediction[i][1] + prediction_diff[i][1]
            if mix >= 1:
                result = 1
            else:
                result = 0
            if result == test_labels[i]:
                count_s += 1
            mix_predict.append(mix / 2)

        # ----Video-level----#
        prediction_video = merge_video_prediction(mix_predict, test_sv, test_vc)
        count_v = 0
        total_v = len(test_labels_video)
        for i, pd in enumerate(prediction_video):
            if pd >= 0.5:
                result = 1
            else:
                result = 0
            if result == test_labels_video[i]:
                count_v += 1
        
        if if_train:
            file = open("./time=" + local_time + "/evaluation.txt",'w')
            file.write("#-----Parameters------#\r\n")
            file.write("Block size (frames per sample): " + str(BLOCK_SIZE) +"\r\n")
            file.write("Epochs: " + str(EPOCHS1) + ", " + str(EPOCHS2) + "\r\n")
            file.write("Batch size: " + str(BATCH_SIZE) + "\r\n")
            file.write("Dropout rate: " + str(DROPOUT_RATE1) + ", " + str(DROPOUT_RATE2) + ", " + str(DROPOUT_RATE3) + ", " + str(DROPOUT_RATE4) + "\r\n")
            file.write("RNN hidden units: " + str(RNN_UNIT1) + ", " + str(RNN_UNIT2) + "\r\n")
            file.write("#---Parameters End----#\r\n")
            file.write("#----Evaluation  Results----#\r\n")
            file.write("Evaluation (g1) - Acc: {:.4}, Loss: {:.4}".format(acc_g1, loss_g1) + "\r\n")
            file.write("Evaluation (g2) - Acc: {:.4}, Loss: {:.4}".format(acc_g2, loss_g2) + "\r\n")
            file.write("Accuracy (sample-level): " + str(count_s / total_s) + "\r\n")
            file.write("Accuracy (video-level): " + str(count_v / total_v) + "\r\n")
            file.write("#------------End------------#")
            file.close()
        else:
            print("#----Evaluation  Results----#\n")
            print("Evaluation (g1) - Acc: {:.4}, Loss: {:.4}".format(acc_g1, loss_g1) + "\n")
            print("Evaluation (g2) - Acc: {:.4}, Loss: {:.4}".format(acc_g2, loss_g2) + "\n")
            print("Accuracy (sample-level): " + str(count_s / total_s) + "\n")
            print("Accuracy (video-level): " + str(count_v / total_v) + "\n")
            print("#------------End------------#")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training and evaluating of LRNet (Tensorflow 2.x version).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-t', '--train', action='store_true',
                        help="If train the model."
                        )
    parser.add_argument('-e', '--evaluate', action='store_true',
                        help="If evaluate the model."
                        )
    parser.add_argument('-g', '--gpu', action='store_true',
                        help="If use the GPU(CUDA) for training."
                        )
    parser.add_argument('-ep1', '--epochs1', type=int, default=800)
    parser.add_argument('-ep2', '--epochs2', type=int, default=200)
    parser.add_argument('-bs', '--batch_size', type=int, default=2000)
    parser.add_argument('-dr1', '--dropout_rate1', type=float, default=0.25)
    parser.add_argument('-dr2', '--dropout_rate2', type=float, default=0.5)
    parser.add_argument('-dr3', '--dropout_rate3', type=float, default=0.25)
    parser.add_argument('-dr4', '--dropout_rate4', type=float, default=0.5)
    parser.add_argument('-ru1', '--rnn_unit1', type=int, default=68)
    parser.add_argument('-ru2', '--rnn_unit2', type=int, default=68)
    args = parser.parse_args()
    main(args)