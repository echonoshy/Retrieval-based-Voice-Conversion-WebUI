import asyncio
import websockets
import wave
import json

async def send_and_receive_audio(uri, audio_file, output_file):
    async with websockets.connect(uri) as websocket:
        # 打开音频文件
        with wave.open(audio_file, 'rb') as wf:
            # 读取并发送音频数据
            while True:
                data = wf.readframes(1024)
                if not data:
                    break
                await websocket.send(data)
        
        # 关闭音频发送后的状态信息
        await websocket.send(json.dumps({"type": "end"}))

        # 接收转码后的音频数据
        with open(output_file, 'wb') as out_file:
            while True:
                response = await websocket.recv()
                if isinstance(response, str):
                    # 检查是否是 JSON 格式的文本数据
                    if response == "end":
                        break
                    else:
                        print(response)
                else:
                    # 将二进制数据写入文件
                    out_file.write(response)

# 运行客户端
audio_file = "input.wav"
output_file = "output.txt"
uri = "ws://localhost:8000/ws"

asyncio.get_event_loop().run_until_complete(
    send_and_receive_audio(uri, audio_file, output_file)
)
