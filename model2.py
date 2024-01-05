from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, LayerNormalization, Softmax, Lambda, \
    Multiply, Activation, Permute, GlobalAvgPool2D, Add, Concatenate
from keras.layers import MaxPooling2D
from keras.layers import Flatten, Dense, Reshape, Average
from keras.models import Sequential, Model
from keras import layers, Model, initializers
import tensorflow as tf
from tensorflow import keras
import math
import keras.backend as K
from keras.layers import LSTM
from keras.layers import Bidirectional


def _MSC(inputs, filters, name, conv_kernels=[3, 5, 7, 9], conv_groups=[1, 2, 4, 8]):
    assert len(conv_kernels) == len(conv_groups)  # 4==4
    split_num = len(conv_kernels)   # split_num = 4
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1   # -1
    in_dim = K.int_shape(inputs)   # 计算输入特征图的通道数
    # print(in_dim)  # (none,8,9,8)
    split_channel = filters // len(conv_kernels)  # 64/4=16   filters=64
    mult_scals = []
    feature_se = []
    for i,  kernel_group in enumerate(zip(conv_kernels, conv_groups)):  # i为索引,kernel_group为conv_kernels和con_groups的值
        kernel, group = kernel_group
        feature = _group_conv(inputs, split_channel, kernel, 1, group)  # group convolution
        feature_se.append(CBAM_attention(feature))
    feature_attention = Concatenate(axis=channel_axis)(feature_se)  # 进行拼接
    return feature_attention

def _group_conv(x, filters, kernel, stride, groups, padding='same'):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    # print(channel_axis) # -1
    in_channels = K.int_shape(x)[channel_axis]
    nb_ig = in_channels // groups
    nb_og = filters // groups
    assert in_channels % groups == 0
    assert filters % groups == 0
    assert filters > groups
    gc_list = []
    for i in range(groups):  # 1,2,4,8
        if channel_axis == -1:
            x_group = Lambda(lambda z: z[:, :, :, i * nb_ig: (i + 1) * nb_ig])(x)
        else:
            x_group = Lambda(lambda z: z[:, i * nb_ig: (i + 1) * nb_ig, :, :])(x)
        gc_list.append(Conv2D(filters=nb_og, kernel_size=kernel, strides=stride,
                              padding=padding, use_bias=False)(x_group))
    return Concatenate(axis=channel_axis)(gc_list) if groups != 1 else gc_list[0]

def create_MSC_CNN(img_size=(8, 9, 8), dropout_rate=0.2, number_of_inputs=1):
    inputs = Input(shape=img_size)
    x1 = _PSA(inputs, 64, "PSA")
    x1 = CBAM_attention(x1)
    x1 = MaxPooling2D(2, 2, name='pool2')(x1)  # (none,4,4,64)
    x1 = BatchNormalization()(x1)  # (none,4,4,64)
    x1 = Dropout(dropout_rate)(x1)
    x = Flatten(name='fla1')(x1)  # (none,1024)
    x = Dense(512, activation='selu', name='dense1')(x)  # (none,512)
    x = Reshape((1, 512), name='reshape1')(x)  # (none,1,512)
    x = LSTM(1024)(x)
    x = BatchNormalization()(x)  # (none,1,512)
    x = Dropout(dropout_rate)(x)  # (none,1,512)
    x = Flatten(name='flat')(x)  # (none,512)
    out_v = Dense(2, activation='softmax', name='out_v')(x)  # (none,2)
    out_a = Dense(2, activation='softmax', name='out_a')(x)  # (none,2)
    model = Model(inputs, [out_v, out_a])
    return model



def CBAM_attention(inputs):
    x = channel_attenstion(inputs)
    x = spatial_attention(x)
    return x

def channel_attenstion(inputs, ratio=0.25):
    channel = inputs.shape[-1]
    x_max = layers.GlobalMaxPooling2D()(inputs)
    x_avg = layers.GlobalAveragePooling2D()(inputs)
    x_max = layers.Reshape([1, 1, -1])(x_max)
    x_avg = layers.Reshape([1, 1, -1])(x_avg)
    x_max = layers.Dense(channel * ratio)(x_max)
    x_avg = layers.Dense(channel * ratio)(x_avg)
    x_max = layers.Activation('relu')(x_max)
    x_avg = layers.Activation('relu')(x_avg)
    x_max = layers.Dense(channel)(x_max)
    x_avg = layers.Dense(channel)(x_avg)
    x = layers.Add()([x_max, x_avg])
    x = tf.nn.sigmoid(x)
    x = layers.Multiply()([inputs, x])  # [h,w,c]*[1,1,c]==>[h,w,c]
    return x

def spatial_attention(inputs):
    x_max = tf.reduce_max(inputs, axis=3, keepdims=True)  # 在通道维度求最大值
    x_avg = tf.reduce_mean(inputs, axis=3, keepdims=True)  # axis也可以为-1
    x = layers.concatenate([x_max, x_avg])
    x = layers.Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding='same')(x)
    x = tf.nn.sigmoid(x)
    x = layers.Multiply()([inputs, x])
    return x


