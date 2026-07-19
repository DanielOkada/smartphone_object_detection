import * as ort from "onnxruntime-web";
import { generateGridAnchors, postProcess, nms } from "./postprocess";
import { drawDetections } from "./draw";

const INPUT_SIZE = 320;

//==============================
// Canvas
//==============================

// 表示用
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// 推論用（非表示）
const videoCanvas = document.getElementById("videoCanvas");
videoCanvas.width = INPUT_SIZE;
videoCanvas.height = INPUT_SIZE;
const videoCtx = videoCanvas.getContext("2d");

//==============================
// モデル
//==============================

const session = await ort.InferenceSession.create(
    `${import.meta.env.BASE_URL}model/nanodet.onnx`,
);
const anchors = generateGridAnchors(INPUT_SIZE, INPUT_SIZE);

//==============================
// COCOクラス
//==============================

const response = await fetch(`${import.meta.env.BASE_URL}coco_names.json`);
const classNames = await response.json();

//==============================
// FPS
//==============================

let frameCount = 0;
let lastTime = performance.now();
let fps = 0;

function updateFps() {
    frameCount++;

    const now = performance.now();
    const elapsed = now - lastTime;

    if (elapsed >= 1000) {
        fps = (frameCount * 1000) / elapsed;
        frameCount = 0;
        lastTime = now;
    }

    return fps;
}

//==============================
// preprocess
//==============================

function preprocess(imageData) {
    const { data, width, height } = imageData;

    const input = new Float32Array(3 * width * height);
    const channelSize = width * height;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const pixel = (y * width + x) * 4;
            const index = y * width + x;

            input[index] = data[pixel] / 255.0;
            input[channelSize + index] = data[pixel + 1] / 255.0;
            input[channelSize * 2 + index] = data[pixel + 2] / 255.0;
        }
    }

    return input;
}

//==============================
// log
//==============================

const logElement = document.getElementById("log");

function log(...args) {
    console.log(...args);

    logElement.textContent +=
        args
            .map((x) =>
                typeof x === "object" ? JSON.stringify(x, null, 2) : String(x),
            )
            .join(" ") + "\n";

    logElement.scrollTop = logElement.scrollHeight;
}

window.onerror = (message, source, line, column, error) => {
    log("=== ERROR ===");
    log(message);

    if (error?.stack) {
        log(error.stack);
    }
};

//==============================
// Camera
//==============================

const video = document.getElementById("video");

const stream = await navigator.mediaDevices.getUserMedia({
    video: {
        facingMode: "environment",
    },
    audio: false,
});

video.srcObject = stream;

await new Promise((resolve) => {
    video.onloadedmetadata = resolve;
});

await video.play();

// 表示Canvasはカメラサイズに合わせる
canvas.width = video.videoWidth;
canvas.height = video.videoHeight;

//==============================
// Detect
//==============================

async function detect() {
    //---------- 推論用 ----------
    videoCtx.drawImage(video, 0, 0, INPUT_SIZE, INPUT_SIZE);

    const imageData = videoCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);

    const inputData = preprocess(imageData);

    const tensor = new ort.Tensor("float32", inputData, [
        1,
        3,
        INPUT_SIZE,
        INPUT_SIZE,
    ]);

    const start = performance.now();

    const outputs = await session.run({
        data: tensor,
    });

    const inferenceTime = performance.now() - start;

    const detections = postProcess(
        outputs.output.cpuData,
        anchors,
        INPUT_SIZE,
        INPUT_SIZE,
    );

    const results = nms(detections);

    //---------- 表示 ----------
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const scaleX = canvas.width / INPUT_SIZE;
    const scaleY = canvas.height / INPUT_SIZE;

    drawDetections(ctx, results, classNames, scaleX, scaleY);

    ctx.font = "20px sans-serif";

    ctx.fillStyle = "lime";
    ctx.fillText(`FPS: ${updateFps().toFixed(1)}`, 10, 25);

    ctx.fillStyle = "white";
    ctx.fillText(`Inference: ${inferenceTime.toFixed(1)} ms`, 10, 50);

    requestAnimationFrame(detect);
}

detect();
