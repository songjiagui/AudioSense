
import pyaudio
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from funasr import AutoModel
import soundfile as sf
import asyncio
import time
import re  # 导入正则表达式模块
import os  # 在文件开头导入 os 模块，此处已有导入，可不重复添加
from flask import Flask, Response
import threading
from flask_cors import CORS
from queue import Queue  # 导入 Queue
import json  # 导入 json 模块

app = Flask(__name__)
CORS(app)  # 处理跨域问题
sse_clients = []

# 优化函数：发送 SSE 消息
def send_sse_message(data):
    """
    向所有连接的 SSE 客户端发送消息。

    :param data: 要发送的数据，应为字典类型
    """
    try:
        json_data = json.dumps(data, ensure_ascii=False)
        for client_queue in sse_clients[:]:  # 使用切片避免修改原列表
            try:
                client_queue.put(json_data)
            except Exception as e:
                print(f"发送 SSE 消息出错: {e}")
                sse_clients.remove(client_queue)
    except Exception as e:
        print(f"JSON 序列化出错: {e}")

@app.route('/stream')
def stream():
    from queue import Queue
    client_queue = Queue()
    sse_clients.append(client_queue)

    def event_stream():
        try:
            while True:
                try:
                    # 从队列获取消息，设置超时时间避免阻塞
                    message = client_queue.get(timeout=1)
                    yield f"data: {message}\n\n"
                except Exception:
                    # 没有消息时发送心跳，心跳也用 JSON 格式
                    heartbeat = json.dumps({"status": "heart beat"}, ensure_ascii=False)
                    yield f"data: {heartbeat}\n\n"
                time.sleep(1)
        except GeneratorExit:
            if client_queue in sse_clients:
                sse_clients.remove(client_queue)

    return Response(event_stream(), mimetype='text/event-stream')

class Config:
    """
    配置类，包含语音识别和音频处理的相关参数。
    """
    chunk_size_ms: int = 300  # 每个音频块的大小，单位为毫秒 300
    sample_rate: int = 16000
    bit_depth: int = 16
    channels: int = 1
    avg_logprob_thr: float = -0.25
    sv_thr: float = 0.42  # 声纹验证阈值
    # 自动获取 user 文件夹下的文件
    reg_spks_files: list = [os.path.join("user", f) for f in os.listdir("user") if os.path.isfile(os.path.join("user", f))]  # 注册用户的音频文件列表

config = Config()
print(f"reg_spks_files: {config.reg_spks_files}")
# asr_pipeline = pipeline(
#     task=Tasks.auto_speech_recognition,
#     model='/iic/SenseVoiceSmall',
#     model_revision="master",
#     device="cuda:0",
#     ban_emo_unk=True,
#     disable_update=True
# )
sv_pipeline = pipeline(
            task='speaker-verification',
            # model='/iic/speech_eres2net_large_200k_sv_zh-cn_16k-common',
            # model_revision='v1.0.0',
            model='/iic/speech_campplus_sv_zh-cn_16k-common',
            model_revision='v1.0.0',
            device='cuda:0',
            ngpu=1
        )
model_asr = AutoModel(
    model="/iic/SenseVoiceSmall",
    trust_remote_code=True,
    remote_code="./model.py",
    device="cuda:0",
    disable_update=True
)

model_vad = AutoModel(
    model="/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    model_revision="v2.0.4",
    disable_pbar=True,
    max_end_silence_time=500,
    speech_noise_thres=0.6,
    device="cuda:0",
    disable_update=True
)

def asr(audio, lang="auto", cache=None, use_itn=True):
    """
    进行语音识别。

    :param audio: 音频数据
    :param lang: 语言类型，默认为自动检测
    :param cache: 缓存数据，默认为 None
    :param use_itn: 是否使用 ITN（Inverse Text Normalization），默认为 False
    :return: 语音识别结果
    """
    if cache is None:
        cache = {}
    start_time = time.time()
    result = model_asr.generate(
        input=audio,
        cache=cache,
        language=lang.strip(),
        use_itn=use_itn,
        batch_size_s=60,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"asr elapsed: {elapsed_time * 1000:.2f} milliseconds")
    return result


async def realtime_transcription():
    """
    异步函数，实时从麦克风获取音频输入，进行语音活动检测和语音识别。
    """
    
    audio_verify_buffer = np.array([], dtype=np.float32)

    def _reg_spk_init(files):
        """
        初始化注册用户的声纹特征。

        :param files: 注册用户的音频文件列表。
        :return: 包含注册用户声纹特征的字典。
        """
        reg_spk = {}
        for f in files:
            try:
                data, sr = sf.read(f, dtype="float32")
                if sr != config.sample_rate:
                    print(f"音频文件 {f} 的采样率 {sr} 与设定采样率 {config.sample_rate} 不一致，可能影响识别结果。")
                k, _ = os.path.splitext(os.path.basename(f))
                reg_spk[k] = {
                    "data": data,
                    "sr": sr,
                }
            except Exception as e:
                print(f"注册用户声纹特征时出错: {e}")
        return reg_spk

    def verify_speaker(audio, clean_text):
        """
        进行声纹验证。

        :param audio: 待验证的音频数据。
        :return: 是否为注册用户，以及对应的用户名（如果是）。
        """
        # 从配置类中获取常量
        sv_thr = config.sv_thr
        reg_spks_files = config.reg_spks_files
        reg_spks = _reg_spk_init(reg_spks_files)
        for k, v in reg_spks.items():
            try:
                res_sv = sv_pipeline([audio, v["data"]], sv_thr)
                if res_sv["score"] >= sv_thr:
                    print(f"验证通过，用户: {k}, 得分: {res_sv['score']}")
                    message = {"status": "success", "user": k, "message": clean_text}
                    send_sse_message(message)
                    return True, k
            except Exception as e:
                print(f"声纹验证出错: {e}")
        print("验证未通过。")
        message = {"status": "error","user": "", "message": clean_text}
        send_sse_message(message)
        return False, None

    p = pyaudio.PyAudio()
    chunk_size = int(config.chunk_size_ms * config.sample_rate / 1000)
    print(f"chunk_size: {chunk_size}")
    stream = p.open(format=pyaudio.paInt16,
                    channels=config.channels,
                    rate=config.sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    audio_buffer = np.array([], dtype=np.float32)
    audio_vad = np.array([], dtype=np.float32)
    cache = {}
    cache_asr = {}
    last_vad_beg = last_vad_end = -1
    offset = 0

    print("麦克风已就绪，开始说话吧...")
    try:
        while True:
            data = stream.read(chunk_size)
            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
            audio_buffer = np.append(audio_buffer, audio_chunk)

            while len(audio_buffer) >= chunk_size:
                chunk = audio_buffer[:chunk_size]
                audio_buffer = audio_buffer[chunk_size:]
                audio_vad = np.append(audio_vad, chunk)

                res = model_vad.generate(
                    input=chunk,
                    cache=cache,
                    is_final=False,
                    chunk_size=config.chunk_size_ms
                )
                if len(res[0]["value"]):
                    vad_segments = res[0]["value"]
                    for segment in vad_segments:
                        if segment[0] > -1:
                            last_vad_beg = segment[0]
                        if segment[1] > -1:
                            last_vad_end = segment[1]
                        if last_vad_beg > -1 and last_vad_end > -1:
                            last_vad_beg -= offset
                            last_vad_end -= offset
                            offset += last_vad_end
                            beg = int(last_vad_beg * config.sample_rate / 1000)
                            end = int(last_vad_end * config.sample_rate / 1000)
                            print(f"vad beg: {beg}, end: {end}")
                            result = asr(audio_vad[beg:end], "auto", cache_asr, True)
                            if result is not None and result[0]['text']:
                                # 使用正则表达式移除标签
                                clean_text = re.sub(r'<\|[^>]*\|>', '', result[0]['text'])
                                print(f"识别结果: {clean_text}")
                            # 更新验证缓冲区
                            new_size = len(audio_verify_buffer) + len(audio_vad[beg:end])
                            audio_verify_buffer = np.resize(audio_verify_buffer, new_size)
                            audio_verify_buffer[-len(audio_vad[beg:end]):] = audio_vad[beg:end]

                            # 调用声纹验证
                            verify_speaker(audio_verify_buffer, clean_text)
                            audio_verify_buffer = np.array([], dtype=np.float32)  # 清空缓冲区
                            audio_vad = audio_vad[end:]
                            last_vad_beg = last_vad_end = -1

    except KeyboardInterrupt:
        print("程序已停止。")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
def run_flask_app():
    app.run(host='0.0.0.0', port=5000)
@app.route('/stop', methods=['POST'])
def stop_verification():
    # 这里暂时没有实际的停止逻辑，可根据需求完善
    return json.dumps({"status": "success", "message": "实时声纹识别已停止"}), 200

if __name__ == "__main__":
    # 启动 Flask 服务器
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.daemon = True
    flask_thread.start()

    # 使用 asyncio 运行异步函数
    async def main():
        await realtime_transcription()

    # 创建事件循环并运行主函数
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())