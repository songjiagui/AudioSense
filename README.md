# AudioSense

这是一个带声纹识别的实时语音转文字的程序。
主要使用speech_fsmn_vad_zh-cn-16k-common-pytorch进行音频切割，SenseVoiceSmall进行转文字，speech_campplus_sv_zh-cn_16k-common进行声纹识别。

安装
pip install -r requirements.txt
在同级目录下新建一个user目录，来存放识别的样本。
