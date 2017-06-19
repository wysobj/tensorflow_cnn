import os
import pickle
import numpy as np

def load_cifar100(data_set="train", reshape=False, subtract_avg=True, normalize=False):
    path = os.path.join("data100", data_set)
    images, labels = load_file(path, reshape, normalize, 100)
    if subtract_avg:
        mean = images.mean(axis=0)
        print(mean)
        images = images - mean
    return np.array(images), np.array(labels)

def load_cifar10(data_set="train", reshape=False, subtract_avg=True, normalize=False):
    train_data_files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    test_data_files = ["test_batch"]
    load_files = train_data_files if data_set == "train" else test_data_files
    images = []
    labels = []
    for fn in load_files:
        path = os.path.join("data10", fn)
        images_fn, labels_fn = load_file(path, reshape, normalize, 10)
        images.extend(images_fn)
        labels.extend(labels_fn)
    images = np.array(images)
    if subtract_avg:
        mean = images.mean(axis=0)
        images = images - mean
    return images, np.array(labels)

def load_file(path, reshape, normalize, label_size):
    data = pickle.load(open(path, "rb"), encoding="bytes")
    images = data[b"data"]
    if normalize:
        images = images / 255
    labels = data[b"labels"]
    if reshape:
        images = [img.reshape((3, 32, 32)).transpose(1, 2, 0) for img in images]
        images = np.array(images)
    one_hot_labels = []
    for label in labels:
        v = np.zeros(label_size)
        v[label] = 1
        one_hot_labels.append(v)
    labels = one_hot_labels
    return images, labels

def split_data(X, y, proportion=0.9):
    size = len(X)
    indexes = np.arange(size)
    split_idx = int(size * proportion)
    np.random.shuffle(indexes)
    train_indexes = indexes[ : split_idx]
    valid_indexes = indexes[split_idx : ]
    train_X, train_y = X[train_indexes], y[train_indexes]
    valid_X, valid_y = X[valid_indexes], y[valid_indexes]
    return train_X, train_y, valid_X, valid_y

def img_flip(img):
    return img[ :, : : -1, :]

def img_crop(img, offset_x, offset_y, crop_size):
    return img[ offset_x : offset_x+crop_size, offset_y : offset_y+crop_size, :]

def img_whitening(img):
    return (img - np.mean(img)) / np.std(img)

def rand_flip(img):
    flag = np.random.randint(2)
    if flag == 1:
        return img_flip(img)
    return img

def rand_crop(img, crop_size):
    img_size = len(img)
    offset_boundry = img_size - crop_size + 1
    rand_offset_x = np.random.randint(offset_boundry)
    rand_offset_y = np.random.randint(offset_boundry)
    return img_crop(img, rand_offset_x, rand_offset_y, crop_size)

def rand_transform(img, crop_size):
    cropped = rand_crop(img, crop_size)
    flipped = rand_flip(cropped)
    return flipped

def fix_crop(img, crop_size):
    img_size = len(img)
    max_offset = img_size - crop_size
    middle_offset = int(max_offset / 2)
    img1 = img_crop(img, 0, 0, crop_size)
    img2 = img_crop(img, 0, max_offset, crop_size)
    img3 = img_crop(img, middle_offset, middle_offset, crop_size)
    img4 = img_crop(img, max_offset, 0, crop_size)
    img5 = img_crop(img, max_offset, max_offset, crop_size)
    pieces = [img1, img2, img3, img4, img5]
    return pieces

def batch_rand_transform(imgs, crop_size):
    batch_imgs = []
    for img in imgs:
        batch_imgs.append(img_whitening(rand_transform(img, crop_size)))
    return np.array(batch_imgs)

def batch_fix_crop(imgs, crop_size):
    batch_imgs = []
    for img in imgs:
        batch_imgs.extend(fix_crop(img, crop_size))
    batch_imgs = [img_whitening(img) for img in batch_imgs]
    return np.array(batch_imgs)