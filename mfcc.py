import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import os
import sys
import math
from scipy.fftpack import dct
sys.path.append("..")
from config import cfg

def extract_mfcc(input_signal, sample_rate):
    """
    提取mfcc特征
    将音频wav格式数据生成mfcc格式数据，用于模型训练
    :param:input_signal  wav音频数据格式数组
    :param:sample_rate   采样率
    """

    # 1、预加重 y(t)=x(t)−αx(t−1)
    emphasized_signal = np.append(input_signal[0], input_signal[1:] - cfg.mfcc.pre_emphasis * input_signal[:-1])

    # 2、分帧 CHUNK
    frame_size = cfg.mfcc.frame_length_ms * 0.001   # 毫秒转成秒 frame_size = 32ms * 0.001 = 0.032s
    frame_stride = cfg.mfcc.frame_shift_ms * 0.001  # 毫秒转成秒 frame_stride = 20ms * 0.001 = 0.020s
    # sample_rate = 8000    采样率 一秒钟采集多少数据
    # frame_size = 0.032    帧长 单位秒
    # frame_stride = 0.020  帧移 单位秒
    # frame_length = frame_size * sample_rate = 0.032 * 8000 = 256 帧长(数据量) 单位个
    # frame_step = frame_stride * sample_rate = 0.020 * 8000 = 160 帧移(数据量) 单位个
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
    # TODO 核实是否是 8000
    # 音频长度 8000
    signal_length = len(emphasized_signal)
    # 帧长 250
    # round() 四舍五入值
    frame_length = int(round(frame_length))
    # 帧移 160
    frame_step = int(round(frame_step))
    # 总帧数 按帧移160计算
    # num_frames = |8000 - 256| / 160 + 1 = 49
    # abs() 绝对值
    # eil() 向上取整
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step + 1))

    # 49 * 160 + 256 = 8096
    pad_signal_length = num_frames * frame_step + frame_length
    # 填充零 8096 - 8000 = 96
    z = np.zeros((pad_signal_length - signal_length))
    # 填充零后的数据
    pad_signal = np.append(emphasized_signal, z)

    # tile() 函数，就是将原矩阵横向、纵向地复制。
    # num_frames * frame_step = 49 * 160 = 8096
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T


    # 用矩阵来表示帧的次数
    # TODO 核查是否 49*250
    # 49*256，49:总的帧数，250:每一帧的采样数
    # 第一帧采样为0、1、2...200;第二帧为80、81、81...280..依次类推
    # frame：348*200，横坐标348为帧数，即时间；纵坐标200为一帧的200毫秒时间，内部数值代表信号幅度
    """
        frames == = [[0.00000000e+00 - 4.17850577e-05 - 4.17850577e-05... - 4.17850577e-05
                  4.17850577e-05  8.35701153e-05]
                 [0.00000000e+00  0.00000000e+00  0.00000000e+00...  0.00000000e+00
                 0.00000000e+00 - 4.17850577e-05]
    """
    frames = pad_signal[np.mat(indices).astype(np.int32, copy=False)]

    # 3、加汉明窗
    # 傅里叶变换默认操作的时间段内前后端点是连续的，即整个时间段刚好是一个周期，
    # 但是，显示却不是这样的。所以，当这种情况出现时，仍然采用FFT操作时，
    # 就会将单一频率周期信号认作成多个不同的频率信号的叠加，而不是原始频率，这样就差生了频谱泄漏问题
    frames *= np.hamming(frame_length) # 相乘，和卷积类似
    # frames *= 0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1)) # Explicit Implementation **

    # 4、傅里叶变换 np
    # 傅立叶变换和功率谱
    # 将49*256，扩展成49*256
    NFFT = 256
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT (FFT的幅值)
    # print(mag_frames.shape)
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))  # Power Spectrum (功率谱)

    # 5、三角滤波
    low_freq_mel = 0
    # 将频率转换为Mel
    nfilt = 40  # mel滤波器组：40个滤波器
    # high_freq_mel = (2595 * math.log10(6.714285))
    high_freq_mel = (2595 * math.log10(1 + (sample_rate / 2) / 700))  # # Convert Hz to Mel 2146
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz

    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))

    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    # mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')
    # 用离散余弦变换（DCT）对滤波器组系数去相关处理，并产生滤波器组的压缩表示
    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # 保持在2-13

    # 将正弦升降1应用于MFCC以降低已被声称在噪声信号中改善语音识别的较高MFCC.
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    cep_lifter = 22
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift
    return np.around(mfcc)


