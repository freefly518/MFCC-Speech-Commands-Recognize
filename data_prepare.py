import os
import scipy.io.wavfile
import random
import numpy as np
import tensorflow as tf
from mfcc import *
# 音频目录
data_dir = "./wav_data/"
# 标签目录
data_label_dir = "./data_label/"

# 训练数据集目录
label_train_txt = "./train_label.txt"
# 验证数据集目录
label_val_txt = "./val_label.txt"

# shuffle 训练集
shuffle_label_train_txt = "./train_label_shuffle.txt"

# 训练集tfrecord目录
tfrecord_train = "./tfrecords/train.tfrecords"
# 验证集 tfrecord目录
tfrecord_val = "./tfrecords/val.tfrecords"


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 采样率16000到8000
def resample_signal_16_to_8(origin_signal, origin_rate, rate=2):
    """
    :param:origin_signal 原始数据,格式数组
    :param:origin_rate   原始采样率 16000
    :param:rate          转化率
    """
    if rate > len(origin_signal):
        print("# num out of length, return arange:", end=" ")
        return origin_signal
    resampled_signal = np.array([], dtype=origin_signal.dtype)
    seg = int(len(origin_signal) / rate) # seg = 8000
    for n in range(seg):
        # rate * n = 2 * [1:8000] 16000的偶数位置数据,相当于数据减半，采样率是8000
        resampled_signal = np.append(resampled_signal, origin_signal[int(rate * n)])
    return int(origin_rate/rate), resampled_signal

# 根据wav_data目录里的音频文件，生成标签数据data_label
# wav_data      格式：文件夹：no, off, on, yes
# data_label    格式：  文件：label_no.txt, label_off.txt, label_on.txt, label_yes.txt
def gen_label_rm_bad():
    wav_dirs = os.listdir(data_dir)  
    for index, wav_dir in enumerate(wav_dirs):
        current_wav_dir = data_dir + wav_dir
        current_label_txt = data_label_dir + "lable_" + wav_dir + ".txt"
        wav_file_names = os.listdir(current_wav_dir)
        with open(current_label_txt, "w") as f:
            for _, wav_file_name in enumerate(wav_file_names):
                current_wav_file = current_wav_dir + "/" + wav_file_name
                sample_rate, signal = scipy.io.wavfile.read(current_wav_file)
                if len(signal) >= 15000 and sample_rate == 16000:
                    file_label = current_wav_file + " " + str(index) + "\n"
                    f.write(file_label)
        # f.close() 不用手动关闭

# 划分数据集
def split_train_val_label():
    label_filenames = os.listdir(data_label_dir) 
    with open(label_train_txt, "w")as fa, open(label_val_txt, "w") as fb:
        for _, label_file in enumerate(label_filenames):
            f1 = data_label_dir + label_file
            with open(f1, "r") as f:
                lines = f.readlines()
            # f.close() 不用手动关闭
            for _ in range(int(len(lines) * 0.8)):
                # lines.pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值
                fa.write(lines.pop(random.randint(0, len(lines) - 1)))
            fb.writelines(lines)
    # fa.close() 不用手动关闭
    # fb.close() 不用手动关闭

# 训练集 shuffle
def train_label_shuffle():
    lines = []
    with open(shuffle_label_train_txt, 'w') as out:
        with open(label_train_txt, 'r') as infile:
            for line in infile:
                lines.append(line)
        # 进行4次shuffle
        random.shuffle(lines)
        random.shuffle(lines)
        random.shuffle(lines)
        random.shuffle(lines)
        for line in lines:
            out.write(line)
    #     infile.close() 不用手动关闭
    # out.close()

# 生成wav、labels文件
def get_imgs_labels(label_file):
    wav_filenames = []
    labels = []
    with open(label_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            wav_filenames.append(line.split()[0])
            labels.append(line.split()[1])
    # f.close() 不用手动关闭
    return wav_filenames, labels

def gennerate_terecord_file(tfrecordfilename, label_file):
    """
    生成 tfrecord 文件
    :param:tfrecordfilename tfrecord文件路径和名称
    :param:label_file label文件
    """
    cnt = 0
    # 获取wav文列表，label列表
    filenames, labels = get_imgs_labels(label_file)
    with tf.io.TFRecordWriter(tfrecordfilename) as writer:
        for filename, label in zip(filenames, labels):
            cnt = cnt + 1
            # sample_rate 频率
            # wave_data 音频数据格式数组
            sample_rate, wave_data = scipy.io.wavfile.read(filename)
            # 采样率16000到8000
            # resampled 采样率
            # resampled_signal 音频数据格式数组
            resampled_rate, resampled_signal = resample_signal_16_to_8(wave_data, sample_rate, rate=2)
            # resampled_signal 少于8000部分用零填充
            resampled_signal = np.append(resampled_signal, np.zeros(8000 - len(resampled_signal), dtype=np.int16))

            mfcc = extract_mfcc(resampled_signal, resampled_rate).astype(np.int64)    

            mfcc_features = np.reshape(mfcc, [50, 12])
            mfcc_features = mfcc_features.tostring()
            label = int(label)
            feature = {  
                'wave_data': _bytes_feature(mfcc_features),  
                'label': _int64_feature(label)  
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))  
            writer.write(example.SerializeToString())  
            if cnt % 100 == 0:
                print("the length of tfrecord is: %d " % cnt)
        print("total : %d " % cnt)


def _parse_example(example_string):
    feature_description = { 
        'wave_data': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    feature_dict = tf.io.parse_single_example(example_string, feature_description)

    wave_data = feature_dict['wave_data']
    wave_data = tf.io.decode_raw(wave_data, tf.int64)
    wave_data = tf.reshape(wave_data, [50, 12, 1])
    
    wave_data =tf.cast(((wave_data - tf.math.reduce_min(wave_data)) / (tf.math.reduce_max(wave_data) - tf.math.reduce_min(wave_data))), tf.float32)

    labels = feature_dict['label']
    labels = tf.cast(labels, tf.int64)

    return wave_data, labels


def gen_data_batch(file_pattern, batch_size, num_repeat=5, is_training=True):
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    if is_training:
        dataset = dataset.repeat(num_repeat)
        dataset = dataset.map(_parse_example, num_parallel_calls=4)
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(buffer_size=16 * batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(_parse_example, num_parallel_calls=4)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


if __name__ == '__main__':
    ## Execute sequentially and everyyime only one
    # gen_label_rm_bad()        # 生成label
    # split_train_val_label()   # 划分数据集
    # train_label_shuffle()     # shuffle
    # gennerate_terecord_file(tfrecord_train, shuffle_label_train_txt)  # 生成tfrecord_train
    # gennerate_terecord_file(tfrecord_val, label_val_txt)              # 生成tfrecord_val

    # For test the save information of tfrecord files is OK or not
    dataset1 = gen_data_batch(tfrecord_val, 6, 2, is_training=False)
    for batch, (wave_data_i, labels_i) in enumerate(dataset1):
        for k in range(7):
            wave_data_ = wave_data_i[k]
            lable = labels_i[k].numpy()
            wavname = str(k) + "_" + str(lable) + ".wav"
            scipy.io.wavfile.write(wavname, 8000, np.squeeze(wave_data_))
    print("<<<<")