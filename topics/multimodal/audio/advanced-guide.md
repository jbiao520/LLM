# 音频处理深入版

> 面向有 ML 基础读者的音频处理深度指南

## 1. 音频特征提取

### 1.1 短时傅里叶变换 (STFT)

$$X(m, \omega) = \sum_{n=-\infty}^{\infty} x[n] w[n-m] e^{-j\omega n}$$

其中 $w[n]$ 是窗函数（如 Hamming 窗）。

### 1.2 梅尔频谱

梅尔频率:
$$m = 2595 \log_{10}(1 + \frac{f}{700})$$

逆变换:
$$f = 700 (10^{m/2595} - 1)$$

梅尔滤波器组:
```python
def mel_filterbank(n_filters, n_fft, sample_rate):
    mel_min = hz_to_mel(0)
    mel_max = hz_to_mel(sample_rate / 2)

    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points = mel_to_hz(mel_points)

    filterbank = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(n_filters):
        filterbank[i] = triangular_filter(hz_points[i:i+3])

    return filterbank
```

## 2. Whisper 架构详解

### 2.1 预处理

```python
def preprocess_audio(audio, sample_rate=16000):
    # 重采样到 16kHz
    audio = resample(audio, sample_rate, 16000)

    # 填充/截断到 30 秒
    target_length = 30 * 16000
    if len(audio) < target_length:
        audio = pad(audio, target_length)
    else:
        audio = audio[:target_length]

    # 计算 Log-Mel 频谱
    mel = melspectrogram(audio, n_mels=80, n_fft=400, hop_length=160)
    mel = torch.log(mel + 1e-8)

    return mel  # (80, 3000)
```

### 2.2 模型结构

```
┌─────────────────────────────────────────────────────────────────┐
│                     Whisper 模型结构                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Encoder:                                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Conv1d (3, 80, 3, padding=1)                           │   │
│   │  Conv1d (3, 80, 3, stride=2)  × 2                       │   │
│   │  Positional Embedding                                   │   │
│   │  Transformer Blocks × N                                 │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   Decoder:                                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Token Embedding                                        │   │
│   │  Positional Embedding                                   │   │
│   │  Transformer Blocks × N                                 │   │
│   │  Cross-Attention to Encoder                             │   │
│   │  Output Projection                                      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   模型变体:                                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ tiny    : 39M  params, 4 enc, 4 dec                    │   │
│   │ base    : 74M  params, 6 enc, 6 dec                    │   │
│   │ small   : 244M params, 12 enc, 12 dec                  │   │
│   │ medium  : 769M params, 24 enc, 24 dec                  │   │
│   │ large   : 1550M params, 32 enc, 32 dec                 │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 多任务 Token

Whisper 使用特殊的 Token 来控制任务：

```
<|startoftranscript|>  开始
<|en|>                 语言标记
<|transcribe|>         转录任务
<|translate|>          翻译任务（翻译成英文）
<|notimestamps|>       无时间戳
<|timestamp|>          带时间戳
...
<|endoftranscript|>    结束
```

## 3. 现代语音识别

### 3.1 Wav2Vec 2.0

自监督预训练：

```
预训练:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   原始音频 ───▶ CNN Encoder ───▶ 量化 ───▶ 对比学习             │
│                    │                                             │
│                    └──▶ Masked Prediction                        │
│                                                                 │
│   目标: 在 Masked 位置预测正确的量化编码                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

微调:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   预训练 Encoder ───▶ CTC Decoder ───▶ 文本                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 CTC Loss

Connectionist Temporal Classification:

$$P(l|x) = \sum_{\pi \in B^{-1}(l)} \prod_{t=1}^{T} p_t(\pi_t|x)$$

解决输入输出长度对齐问题。

## 4. 语音合成

### 4.1 VITS

端到端 TTS，结合 VAE 和 GAN：

```
┌─────────────────────────────────────────────────────────────────┐
│                     VITS 架构                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   文本 ───▶ 文本编码器 ───▶ 随机分布 (VAE) ───▶ 声码器 ───▶ 音频│
│                               │                                 │
│                               ▼                                 │
│                        后验编码器 ◀── 音频 (训练时)             │
│                                                                 │
│   判别器 ◀─── 生成的/真实的音频 ───▶ GAN Loss                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 声码器 (Vocoder)

| 声码器 | 特点 |
|--------|------|
| WaveNet | 自回归，高质量，慢 |
| HiFi-GAN | GAN-based，快速，高质量 |
| BigVGAN | 改进 HiFi-GAN，更好 |
| Encodec | 神经音频编解码 |

## 5. 语音克隆

### 5.1 说话人嵌入

```python
class SpeakerEncoder(nn.Module):
    def __init__(self):
        self.lstm = nn.LSTM(80, 256, 3, batch_first=True)
        self.projection = nn.Linear(256, 256)

    def forward(self, mel):
        # mel: (batch, time, 80)
        _, (hidden, _) = self.lstm(mel)
        embedding = self.projection(hidden[-1])
        return F.normalize(embedding, dim=-1)
```

### 5.2 零样本语音克隆

```
参考音频 (3秒) ───▶ 说话人编码器 ───▶ 说话人嵌入
                                           │
文本 ───▶ 文本编码器 ───▶ 声学模型 ───(+ embedding)──▶ 频谱图 ───▶ 声码器 ───▶ 音频
```

## 6. 评估指标

### ASR

| 指标 | 定义 |
|------|------|
| WER (Word Error Rate) | $\frac{S + D + I}{N}$ |
| CER (Character Error Rate) | 字符级别的 WER |

### TTS

| 指标 | 定义 |
|------|------|
| MOS (Mean Opinion Score) | 人工评分 1-5 |
| MCD (Mel Cepstral Distortion) | 频谱距离 |
| FID | 生成质量 |

## 参考文献

- [Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)](https://arxiv.org/abs/2212.04356)
- [wav2vec 2.0](https://arxiv.org/abs/2006.11477)
- [VITS: Conditional Variational Autoencoder](https://arxiv.org/abs/2106.06103)
- [HiFi-GAN](https://arxiv.org/abs/2010.05646)
