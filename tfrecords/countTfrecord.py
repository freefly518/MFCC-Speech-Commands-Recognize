import tensorflow as tf

# 计算tfrecord 总数据量
tf_records_filenames = 'train.tfrecords'
# c = 0
# for record in tf.compat.v1.python_io.tf_record_iterator(tf_records_filenames):
#     c += 1
# print(c)

# train.tfrecords   7017
# val.tfrecords     1757

# 21431
# 5362