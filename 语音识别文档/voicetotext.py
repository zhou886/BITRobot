import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import fft
import os


# 一.获取信号的时频图
def compute_fbank(file):
    x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))  # 汉明窗
    fs, wavsignal = wav.read(file)  ##单声道是一维的，立体音是二维
    # wav波形 加时间窗以及时移10ms
    time_window = 25  # 单位ms
    window_length = fs / 1000 * time_window  # 计算窗长度的公式，目前全部为400固定值
    wav_arr = np.array(wavsignal)
    wav_length = len(wavsignal)
    range0_end = int(len(wavsignal) / fs * 1000 - time_window) // 10  # 计算循环终止的位置，也就是最终生成的窗数
    print(range0_end)
    data_input = np.zeros((range0_end, 200), dtype=np.float64)  # 用于存放最终的频率特征数据
    # np.zeros((a,b), dtype = np.float) 输出一个a行b列的float类型的矩阵
    data_line = np.zeros((1, 400), dtype=np.float64)
    for i in range(0, range0_end):
        p_start = i * 160
        p_end = p_start + 400
        data_line = wav_arr[p_start:p_end]
        data_line = data_line * w  # 加窗
        data_line = np.abs(fft(data_line))
        data_input[i] = data_line[0:200]  # 设置为400除以2的值（即200）是取一半数据，因为是对称的
    data_input = np.log(data_input + 1)
    # data_input = data_input[::]
    return data_input


import matplotlib.pyplot as plt

filepath = 'test1.wav'

a = compute_fbank(filepath)
plt.imshow(a.T, origin='lower')
# imshow(X, camp, origin) X可以是数组和图片，camp的值是控制颜色，
# origin坐标轴样式有'lower'坐标原点在左上角和'upper'坐标原点在左下角两种
plt.show()



# 二.数据处理
# 1.获取训练音频文件及标注文件列表
def source_get(source_file):
    train_file = source_file + '\\data'
    label_lst = []
    wav_lst = []
    for root, dirs, files in os.walk(train_file):
        # os.walk(file)是文件、目录遍历器，
        # root 所指的是当前正在遍历的这个文件夹的本身的地址
        # dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
        # files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
        print("====================")
        print("现在的目录：" + root)
        #print("该目录下包含的子目录：" + str(dirs))
        #print("该目录下包含的文件：" + str(files))
        for file in files:
            if file.endswith('.wav') or file.endswith('.WAV'):
                # file.endswith('.wav')判断文件是否以.wav结尾，返回值为布尔
                wav_file = os.sep.join([root, file])
                # os.sep为了解决不同平台上文件路径分隔符差异问题,os.sep.join([a,b])将a和b以文件分隔符分开，a\b
                label_file = wav_file + '.trn'
                wav_lst.append(wav_file)
                label_lst.append(label_file)
    return label_lst, wav_lst


# 读取的每一个sample的时间轴长都不一样，所以需要对时间轴进行处理，选择batch内最长的那个时间为基准，进行padding。
source_file = 'C:\\Users\\lenovo\\Desktop\\data_thchs30'

label_lst, wav_lst = source_get(source_file)

print(label_lst[:10])
print(wav_lst[:10])

# 确认相同id对应的音频文件和标签文件相同
for i in range(10000):
    wavname = (wav_lst[i].split('/')[-1]).split('.')[0]
    # a.split('/')a中的元素见到/符号就分隔开，返回一个list
    labelname = (label_lst[i].split('/')[-1]).split('.')[0]
    if wavname != labelname:
        print('error')


# 2.label数据处理
# 读取音频文件对应的拼音label
def read_label(label_file):
    # 读取文件内容
    with open(label_file, 'r', encoding='utf8') as f:
        data = f.readlines()
        return data[1]


print(read_label(label_lst[0]))


# 输出第一个文件内容

def gen_label_data(label_lst):
    label_data = []
    for label_file in label_lst:
        # 迭代文件
        pny = read_label(label_file)
        # 读取一个文件内容
        label_data.append(pny.strip('\n'))
    # pny.strip(‘a')函数可删除pny字符串两端的a字符并返回新的字符串
    return label_data  # 全部文件内容


label_data = gen_label_data(label_lst)
print(len(label_data))


# 为label建立拼音到id的映射，即词典
def mk_vocab(label_data):
    vocab = []
    for line in label_data:
        line = line.split(' ')
        # 将label_data每个元素以空格划分为新list
        for pny in line:
            if pny not in vocab:
                vocab.append(pny)
    vocab.append('_')
    # 打印字典
    return vocab


vocab = mk_vocab(label_data)
print(vocab[:10])
print(len(vocab))


# 读取到的label映射到对应的id
def word2id(line, vocab):
    # b.index(a)返回a在列表b中的索引
    return [vocab.index(pny) for pny in line.split(' ')]


label_id = word2id(label_data[0], vocab)
print(label_data[0])
print(label_id)

# 3.音频文件处理
fbank = compute_fbank(wav_lst[0])  # compute_fbank时频转化的函数在前面已经定义好了
# 由于声学模型网络结构原因（3个maxpooling层），我们的音频数据的每个维度需要能够被8整除。（不理解）
print(fbank.shape)
fbank = fbank[:fbank.shape[0] // 8 * 8, :]
# 第一个音频文件的时频图
print(fbank.shape)
plt.imshow(fbank.T, origin='lower')
plt.show()

# 4.数据生成器
from random import shuffle

# shuffle(list)随机排列list中的元素，无返回值
shuffle_list = [i for i in range(10000)]
# 打乱生成的数列
shuffle(shuffle_list)


# generator(生成器)：带有yield的函数。返回一个迭代对象（iteraable），通过next()方法得到一个迭代值
def get_batch(batch_size, shuffle_list, wav_lst, label_data, vocab):
    # batch_size的信号时频图和标签数据，存放到两个list中去
    for i in range(10000 // batch_size):
        wav_data_lst = []
        label_data_lst = []
        # 开始和结尾间隔batch_size位
        begin = i * batch_size
        end = begin + batch_size
        sub_list = shuffle_list[begin:end]
        # 切片每次取四个元素
        for index in sub_list:
            # 获得随机的音频文件的时频图
            fbank = compute_fbank(wav_lst[index])
            fbank = fbank[:fbank.shape[0] // 8 * 8, :]
            # 获得字典标签（索引）
            label = word2id(label_data[index], vocab)
            wav_data_lst.append(fbank)
            label_data_lst.append(label)
        yield wav_data_lst, label_data_lst
    # 每次返回batch_size个时频图和四个标注文件数据的索引


batch = get_batch(4, shuffle_list, wav_lst, label_data, vocab)
# 每次使用next()就执行到一个 yield 语句就会中断，并返回一个迭代值，下次执行时从 yield 的下一个语句继续执行。
wav_data_lst, label_data_lst = next(batch)
for wav_data in wav_data_lst:
    # 迭代时频图数组
    print(wav_data.shape)
    # plt.imshow(wav_data.T, origin='lower')
    # plt.show()
for label_data in label_data_lst:
    # 迭代四个文件数据的索引
    print(label_data)
# 获得四个时频图的大小
lens = [len(wav) for wav in wav_data_lst]
print(max(lens))
print(lens)


# padding
# 然而，每一个batch_size内的数据有一个要求，就是需要构成成一个tensorflow块，这就要求每个样本数据形式是一样的。
# 除此之外，ctc需要获得的信息还有输入序列的长度。
# 这里输入序列经过卷积网络后，长度缩短了8倍，因此我们训练实际输入的数据为wav_len//8（网络结构导致）。
def wav_padding(wav_data_lst):
    # 获得四个音频文件时频图的长度
    wav_lens = [len(data) for data in wav_data_lst]
    # 取出最大一个
    wav_max_len = max(wav_lens)
    # 将长度缩短8倍
    wav_lens = np.array([leng // 8 for leng in wav_lens])
    # np.zreos(a, b, c, d)：建立一个a行b列的矩阵，其中数组的每一个元素都是c行d列的零矩阵
    # 列为wav_max_len可保证全部数据都能存下且每个时频文件数据存储格式相同
    new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))  # 为什么要这样做？
    for i in range(len(wav_data_lst)):
        # d = list[a, :b, :, c] 指d等于矩阵list的第a行的0~b列元素矩阵中每行的第c列元素
        # 将第i个时频图数据放入第i行的0~wav_data_lst[i].shape[0]列
        new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]
    return new_wav_data_lst, wav_lens


pad_wav_data_lst, wav_lens = wav_padding(wav_data_lst)
print(pad_wav_data_lst.shape)
print(wav_lens)


# 对label进行padding和长度获取，不同的是数据维度不同，且label的长度就是输入给ctc的长度，不需要额外处理
def label_padding(label_data_lst):
    # 获得四个标签文件索引的长度和最长的一个
    label_lens = np.array([len(label) for label in label_data_lst])
    max_label_len = max(label_lens)
    # 建立一个4行max_label_len列的零矩阵
    # 列为max_label_len表示能将全部标签都放入且对其（存储格式相同）
    new_label_data_lst = np.zeros((len(label_data_lst), max_label_len))
    for i in range(len(label_data_lst)):
        # 将第i个标签文件数据的索引赋值给新数组的第i行len(label_data_lst[i])]列
        new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
    return new_label_data_lst, label_lens


pad_label_data_lst, label_lens = label_padding(label_data_lst)
print(pad_label_data_lst.shape)
print(label_lens)


# 用于训练格式的数据生成器
def data_generator(batch_size, shuffle_list, wav_lst, label_data, vocab):
    for i in range(len(wav_lst) // batch_size):
        wav_data_lst = []
        label_data_lst = []
        begin = i * batch_size
        end = begin + batch_size
        # 将打乱的数列每次按顺序取四位
        sub_list = shuffle_list[begin:end]
        for index in sub_list:
            # 提取一个时频图数据
            fbank = compute_fbank(wav_lst[index])
            # 让提取出来的数据存入一个可以整除8的数组且保证全部数据都能存入
            pad_fbank = np.zeros((fbank.shape[0] // 8 * 8 + 8, fbank.shape[1]))
            pad_fbank[:fbank.shape[0], :] = fbank
            # 从字典中获取标签文件的索引
            label = word2id(label_data[index], vocab)
            wav_data_lst.append(pad_fbank)
            label_data_lst.append(label)
        # 取出满足CTC算法的时频图数据和时频图数据文件长度能整除8的长度
        pad_wav_data, input_length = wav_padding(wav_data_lst)
        # 取出标签文件的索引和索引的长度
        pad_label_data, label_length = label_padding(label_data_lst)
        inputs = {'the_inputs': pad_wav_data,
                  'the_labels': pad_label_data,
                  'input_length': input_length,
                  'label_length': label_length,
                  }
        # ctc指向全为0的一个一维数组
        outputs = {'ctc': np.zeros(pad_wav_data.shape[0], )}
        yield inputs, outputs


import keras
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D
from keras.layers import Reshape, Dense, Lambda
# reshape从新定义shape
from keras.optimizers import Adam
#from keras.optimizers import adam_v2
#adam = adam_v2.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# optimizers优化器
from keras import backend as K
from keras.models import Model

# 卷积和池化是为了提取特征，全连接是对特征归类
# ReLU激活函数是对卷积网络引进非线性，单纯的卷积操作是线性的，现实问题基本上是非线性的
from keras.utils import multi_gpu_model

def conv2d(size):
    # 二维卷积Conv2D(filters卷积核个数， kelnel_size卷积核大小，strides卷积步长，
    # padding补0策略‘valid’边界不处理same保留边界卷积结果，activation激活函数，use_bias是否使用偏置项，
    # kelnel_initializer权值初始化)
    return Conv2D(size, (3, 3), use_bias=True, activation='relu',
                  padding='same', kernel_initializer='he_normal')


def norm(x):
    # 对每个batch归一化处理
    return BatchNormalization(axis=-1)(x)


def maxpool(x):
    # 最大池化
    return MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(x)


def dense(units, activation="relu"):
    # 全连接，对上一层的神经元进行全部连接，实现特征的非线性组合
    return Dense(units, activation=activation, use_bias=True,
                 kernel_initializer='he_normal')


# x.shape=(none, none, none)
# output.shape = (1/2, 1/2, 1/2)
def cnn_cell(size, x, pool=True):
    # cnn + cnn + maxpool构成的组合
    x = norm(conv2d(size)(x))
    x = norm(conv2d(size)(x))
    if pool:
        x = maxpool(x)
    return x


# 添加CTC损失函数，由backend引入
def ctc_lambda(args):
    labels, y_pred, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# 搭建cnn+dnn+ctc的声学模型
class Amodel():
    """docstring for Amodel."""

    def __init__(self, vocab_size):
        super(Amodel, self).__init__()
        self.vocab_size = vocab_size
        self._model_init()
        self._ctc_init()
        self.opt_init()

    def _model_init(self):
        self.inputs = Input(name='the_inputs', shape=(None, 200, 1))
        self.h1 = cnn_cell(32, self.inputs)
        self.h2 = cnn_cell(64, self.h1)
        self.h3 = cnn_cell(128, self.h2)
        self.h4 = cnn_cell(128, self.h3, pool=False)
        # 200 / 8 * 128 = 3200
        self.h6 = Reshape((-1, 3200))(self.h4)
        self.h7 = dense(256)(self.h6)
        self.outputs = dense(self.vocab_size, activation='softmax')(self.h7)
        self.model = Model(inputs=self.inputs, outputs=self.outputs)

    def _ctc_init(self):
        self.labels = Input(name='the_labels', shape=[None], dtype='float32')
        self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(name='label_length', shape=[1], dtype='int64')
        self.loss_out = Lambda(ctc_lambda, output_shape=(1,), name='ctc') \
            ([self.labels, self.outputs, self.input_length, self.label_length])
        self.ctc_model = Model(inputs=[self.labels, self.inputs,
                                       self.input_length, self.label_length], outputs=self.loss_out)

    def opt_init(self):
        opt = Adam(lr=0.0008, beta_1=0.9, beta_2=0.999, decay=0.01, epsilon=10e-8)
        # self.ctc_model=multi_gpu_model(self.ctc_model,gpus=2)
        self.ctc_model.compile(loss={'ctc': lambda y_true, output: output}, optimizer=opt)


am = Amodel(1176)
am.ctc_model.summary()

total_nums = 100
batch_size = 16
# batch_size太大会出现内存溢出
batch_num = total_nums // batch_size
epochs = 50
# Batch大小是在更新模型之前处理的多个样本。Epoch数是通过训练数据集的完整传递次数。批处理的大小必须大于或等于1且小于或等于训练数据集中的样本数。
# 假设您有一个包含200个样本（数据行）的数据集，并且您选择的Batch大小为5和1,000个Epoch。
# 这意味着数据集将分为40个Batch，每个Batch有5个样本。每批五个样品后，模型权重将更新。
# 这也意味着一个epoch将涉及40个Batch或40个模型更新。
# 有1000个Epoch，模型将暴露或传递整个数据集1,000次。在整个培训过程中，总共有40,000Batch。

source_file = 'C:\\Users\\lenovo\\Desktop\\data_thchs30'
label_lst, wav_lst = source_get(source_file)
label_data = gen_label_data(label_lst[:100])
vocab = mk_vocab(label_data)
vocab_size = len(vocab)

print(vocab_size)

shuffle_list = [i for i in range(100)]

am = Amodel(vocab_size)

#for k in range(5):
for k in range(epochs):
    print('this is the', k + 1, 'th epochs trainning !!!')
    # shuffle(shuffle_list)
    batch = data_generator(batch_size, shuffle_list, wav_lst, label_data, vocab)
    am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=1)


def decode_ctc(num_result, num2word):
    result = num_result[:, :, :]
    in_len = np.zeros((1), dtype=np.int32)
    in_len[0] = result.shape[1]
    r = K.ctc_decode(result, in_len, greedy=True, beam_width=10, top_paths=1)
    r1 = K.get_value(r[0][0])
    r1 = r1[0]
    text = []
    for i in r1:
        text.append(num2word[i])
    return r1, text


# 测试模型 predict(x, batch_size=None, verbose=0, steps=None)
batch = data_generator(1, shuffle_list, wav_lst, label_data, vocab)
for i in range(10):
    # 载入训练好的模型，并进行识别
    inputs, outputs = next(batch)
    x = inputs['the_inputs']
    y = inputs['the_labels'][0]
    result = am.model.predict(x, steps=1)
    # 将数字结果转化为文本结果
    result, text = decode_ctc(result, vocab)
    print('数字结果： ', result)
    print('文本结果：', text)
    print('原文结果：', [vocab[int(i)] for i in y])
