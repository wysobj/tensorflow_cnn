import numpy as np
import tensorflow as tf
import os
import pickle
import time
import data_process
import models

def eval_precision_and_loss(X, y, model, crop_size):
    total_num = len(X)
    batch_size = 1000
    correct_num = 0
    total_loss = 0
    batch_num = int(total_num / batch_size)
    for i in range(batch_num):
        start_idx = i * batch_size
        end_idx = (i+1) * batch_size
        if end_idx > total_num:
            end_idx = total_num
        eval_X = X[start_idx : end_idx]
        eval_y = y[start_idx : end_idx]
        eval_X_fc = data_process.batch_fix_crop(eval_X, crop_size)
        distributions = model.distribution(eval_X_fc).reshape([batch_size, 5, model.classes_num]).mean(axis=1)
        loss = np.sum(-np.log(distributions[np.arange(batch_size), eval_y.argmax(axis=1)]))
        total_loss += loss
        correct_in_batch = np.sum((distributions.argmax(axis=1) == eval_y.argmax(axis=1)))
        correct_num += correct_in_batch
    precision = correct_num / total_num
    loss = total_loss / total_num
    return precision, loss

if __name__ == "__main__":
    start_time = time.time()
    train_imgs, train_labels = data_process.load_cifar10("train", reshape=True)
    train_X, train_y, valid_X, valid_y = data_process.split_data(train_imgs, train_labels)
    test_X, test_y = data_process.load_cifar10("test", reshape=True)
    meta = pickle.load(open("data100/meta", "rb"), encoding="bytes")
    label_names = meta[b"fine_label_names"]
    records_dir = "records"
    models_dir = "models"
    model_id = "conv2"
    if not os.path.exists(records_dir) or not os.path.isdir(records_dir):
            os.mkdir(records_dir)
    if not os.path.exists(models_dir) or not os.path.isdir(models_dir):
            os.mkdir(models_dir)
    loss_path = os.path.join(records_dir, model_id+"loss")
    precision_path = os.path.join(records_dir, model_id+"precision")
    models_path = os.path.join(models_dir, model_id+"model")
    sess = tf.Session()
    optimizer = tf.train.AdamOptimizer()

    crop_size = 24
    model = models.ConvNet2([crop_size, crop_size, 3], 10, sess, optimizer=optimizer, model_id=model_id)
    saver = tf.train.Saver(model.get_trainable_variables())
    max_epoch = 200
    early_stop = False
    patience = 10
    best_model_valid = 0
    bad_counter = 0
    best_params = None
    loss_trace = {"train" : [], "valid" : [], "test" : []}
    error_trace = {"train" : [], "valid" : [], "test" : []}
    for epoch in range(max_epoch):
        train_X_transform = data_process.batch_rand_transform(train_X, crop_size)
        model.fit(train_X_transform, train_y)
        train_precision, train_loss = eval_precision_and_loss(train_X, train_y, model, crop_size)
        valid_precision, valid_loss = eval_precision_and_loss(valid_X, valid_y, model , crop_size)
        test_precision, test_loss = eval_precision_and_loss(test_X, test_y, model, crop_size)
        print("Precision : " + "train %f, valid %f, test %f"%(train_precision, valid_precision, test_precision))
        loss_trace["train"].append(train_loss)
        error_trace["train"].append(1 - train_precision)
        loss_trace["valid"].append(valid_loss)
        error_trace["valid"].append(1 - valid_precision)
        if early_stop:
            if epoch > patience and valid_precision < best_model_valid:
                bad_counter += 1
            else:
                bad_counter = 0
                best_model_valid = valid_precision
                best_model_train = train_precision
                best_model_test = test_precision
                saver.save(sess, models_path)
            if bad_counter > patience:
                end_time = time.time()
                time_consume = end_time - start_time
                print("Early stop! Best model: train %f, valid %f, test %f. Time consume %f s."%(best_model_train, best_model_valid, best_model_test, time_consume))
                break
    np.savez(loss_path, **loss_trace)
    np.savez(precision_path, **error_trace)    
    saver.save(sess, models_path)

