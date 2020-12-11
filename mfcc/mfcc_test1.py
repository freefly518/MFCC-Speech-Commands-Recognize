import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy.fftpack import dct

# https://blog.csdn.net/YAOHAIPI/article/details/103548674

import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy.fftpack import dct

# 1、读取语音信号
f = wave.open(r"data/teston1.wav", "rb")
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
str_data = f.readframes(nframes)
signal = np.frombuffer(str_data, dtype=np.short)
# signal = np.fromstring(str_data, dtype=np.short)
signal = signal * 1.0 / (max(abs(signal)))  # 归一化
signal_len = len(signal)


print(signal.shape)
print('===signal===',signal)
print('===signal[0]===', signal[0])
print('===signal[1:]===', signal[1:])
print('===signal[:-1]===', signal[:-1])
print('=====signal_len:===', signal_len)
print('===signal[1:]_len==', len(signal[1:]))
print('===signal[:-1]_len===', len(signal[:-1]))

# 2、预加重
# y(t)=x(t)−αx(t−1)
signal_add = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
print('===signal_add_len===', len(signal_add))
print('===signal_add===', signal_add)
# time = np.arange(0, nframes) / 1.0 * framerate
# plt.figure(figsize=(20, 10))
# plt.subplot(2, 1, 1)
# plt.plot(time, signal)
# plt.subplot(2, 1, 2)
# plt.plot(time, signal_add)
# plt.show()

# 3、分帧
wlen = 512
inc = 128
N = 512
if signal_len < wlen:
    nf = 1
else:
    nf = int(np.ceil((1.0 * signal_len - wlen + inc) / inc))
pad_len = int((nf - 1) * inc + wlen)
zeros = np.zeros(pad_len - signal_len)
pad_signal = np.concatenate((signal, zeros))
print("===pad_signal===", pad_signal)
indices = np.tile(np.arange(0, wlen), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc), (wlen, 1)).T
print("===np.tile(np.arange(0, wlen), (nf, 1))===",  np.tile(np.arange(0, wlen), (nf, 1)))
print("===np.tile(np.arange(0, nf * inc, inc), (wlen, 1).T===", np.tile(np.arange(0, nf * inc, inc), (wlen, 1)).T)
print("===indices===", indices)

indices = np.array(indices, dtype=np.int32)
frames = pad_signal[indices]
print("frames===", frames)
print("===frames.shape===", frames.shape)
win = np.hanning(wlen)

m = 24
s = np.zeros((nf, m))
for i in range(nf):  # 帧数
    x = frames[i:i + 1]
    y = win * x[0]
    a = np.fft.fft(y)  # 快速傅里叶变换
    b = np.square(abs(a))  # 求FFT变换结果的模的平方
    mel_high = 1125 * np.log(1 + (framerate / 2) / 700)  # mel最高频率
    mel_point = np.linspace(0, mel_high, m + 2)  # 将mel频率等距离分成m+2个点
    Fp = 700 * (np.exp(mel_point / 1125) - 1)   # 将等距分好的mel频率转换为实际频率
    w = int(N / 2 + 1)
    df = framerate / N
    fr = []
    for n in range(w):  # mel滤波器的横坐标
        frs = int(n * df)
        fr.append(frs)
    melbank = np.zeros((m, w))
    for k in range(m + 1):   # 画mel滤波器
        f1 = Fp[k - 1]  # 三角形左边点的横坐标
        f2 = Fp[k + 1]  # 三角形右边点的横坐标
        f0 = Fp[k]   # 三角形中心点点的横坐标
        n1 = np.floor(f1 / df)
        n2 = np.floor(f2 / df)
        n0 = np.floor(f0 / df)
        for j in range(w):
            if j >= n1 and j <= n0:
                melbank[k - 1, j] = (j - n1) / (n0 - n1)
            if j >= n0 and j <= n2:
                melbank[k - 1, j] = (n2 - j) / (n2 - n0)
        for c in range(w):
            s[i, k - 1] = s[i, k - 1] + b[c:c + 1] * melbank[k - 1, c]
        plt.plot(fr, melbank[k - 1,])

plt.show()
logs = np.log(s)  # 取对数
num_ceps = 12
D = dct(logs, type=2, axis=0, norm='ortho')[:, 1: (num_ceps + 1)]
# print(D)
# print(np.shape(D))
