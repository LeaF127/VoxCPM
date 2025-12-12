let ws;
let audioProcessor;
let audioStream;
let heartbeatInterval;
let isRecording = false;

const USER_ID = "y123456";
const TOKEN = "token12345-1730889600";
const FROM_LAN = "cn";
const TO_LAN = "en";
const ROLE = "0";
const WS_URL = "ws://175.24.179.12:9301/dotcwsasr";
const CHUNK_SIZE = 3200; // 每次发送的音频字节数 (~0.1秒)
const INTERVAL = 100; // 发送间隔（毫秒）

function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = (Math.random() * 16) | 0;
        const v = c === 'x' ? r : (r & 0x3) | 0x8;
        return v.toString(16);
    });
}

function toggleMic() {
    const button = document.getElementById('toggleMicButton');
    if (isRecording) {
        stopMic();
        button.innerHTML = "开始录音";
        button.disabled = false;
    } else {
        button.innerHTML = '停止录音 <span class="loading"></span>';
        button.disabled = true;
        startMic().then(() => {
            button.disabled = false;
        }).catch(() => {
            button.innerHTML = "开始录音";
            button.disabled = false;
        });
    }
    isRecording = !isRecording;
}

function startMic() {
    return new Promise((resolve, reject) => {
        const SESSION_ID = generateUUID(); // 动态生成 SESSION_ID
        const TARGET_URL = `${WS_URL}?lanid=0&userid=${USER_ID}&token=${TOKEN}&fromlan=${FROM_LAN}&tolan=${TO_LAN}&sid=${SESSION_ID}&role=${ROLE}`;
        ws = new WebSocket(TARGET_URL);

        ws.onopen = () => {
            document.getElementById('output').innerText = "WebSocket 连接已建立，开始录音...";
            // 启动心跳机制
            heartbeatInterval = setInterval(() => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send("ping");
                }
            }, 30000); // 每 30 秒发送一次 ping
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                const errCode = data.errCode || "";
                if (errCode === "3") {
                    // 中间结果，忽略
                    return;
                }
                if (errCode === "0" || errCode === "10") {
                    const finalText = data.result || "";
                    const transText = data.trans || "";
                    document.getElementById('output').innerText += `\n识别结果: ${finalText}, 翻译结果: ${transText}`;
                }
            } catch (e) {
                console.error("无法解析服务器返回的数据:", event.data);
            }
        };

        ws.onerror = (error) => {
            console.error("WebSocket 错误:", error);
            document.getElementById('output').innerText = "WebSocket 连接错误！";
        };

        ws.onclose = () => {
            clearInterval(heartbeatInterval); // 停止心跳
            document.getElementById('output').innerText += "\nWebSocket 连接已关闭。";
        };

        navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
            audioStream = stream;
            const audioContext = new AudioContext();
            const source = audioContext.createMediaStreamSource(stream);
            audioProcessor = audioContext.createScriptProcessor(4096, 1, 1);
            source.connect(audioProcessor);
            audioProcessor.connect(audioContext.destination);
            audioProcessor.onaudioprocess = (e) => {
                const audioData = e.inputBuffer.getChannelData(0);
                const int16Array = convertFloat32ToInt16(audioData);
                sendAudioChunks(int16Array);
            };
            resolve();
        }).catch(error => {
            console.error("无法访问麦克风:", error);
            document.getElementById('output').innerText = "无法访问麦克风，请检查权限设置。";
            reject(error);
        });
    });
}

function sendAudioChunks(audioData) {
    if (ws.readyState === WebSocket.OPEN) {
        const totalChunks = Math.ceil(audioData.length / CHUNK_SIZE);
        for (let i = 0; i < totalChunks; i++) {
            const chunk = audioData.slice(i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE);
            ws.send(chunk);
        }
    }
}

function stopMic() {
    if (audioProcessor) {
        audioProcessor.disconnect();
    }
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
    }
    if (ws) {
        ws.close();
    }
    document.getElementById('output').innerText += "\n录音已停止。";
}

function convertFloat32ToInt16(buffer) {
    let l = buffer.length;
    const buf = new Int16Array(l);
    while (l--) {
        buf[l] = Math.min(1, buffer[l]) * 0x7FFF;
    }
    return buf;
}
