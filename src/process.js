import * as ort from "onnxruntime-web";
const imageInput = document.getElementById("imageInput");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
import { generateGridAnchors, postProcess, nms } from "./postprocess";
import { drawDetections } from "./draw";

imageInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const img = new Image();
    img.src = URL.createObjectURL(file);
    await img.decode();

    canvas.width = 320;
    canvas.height = 320;

    ctx.drawImage(img, 0, 0, 320, 320);
});

const session = await ort.InferenceSession.create("/model/nanodet.onnx");
// // 入力名と出力名を確認
// console.log(session.inputNames);
// console.log(session.outputNames);

const detectButton = document.getElementById("detectButton");

function preprocess(imageData) {
    const { data, width, height } = imageData;

    // CHW形式
    const input = new Float32Array(3 * width * height);

    const channelSize = width * height;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const pixelIndex = (y * width + x) * 4;

            const r = data[pixelIndex] / 255.0;
            const g = data[pixelIndex + 1] / 255.0;
            const b = data[pixelIndex + 2] / 255.0;

            const index = y * width + x;

            // CHW
            input[index] = r;
            input[channelSize + index] = g;
            input[channelSize * 2 + index] = b;
        }
    }

    return input;
}

detectButton.addEventListener("click", async () => {
    const response = await fetch("/coco_names.json");
    const classNames = await response.json();

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    const inputData = preprocess(imageData); // ImageData → Float32Array

    const tensor = new ort.Tensor("float32", inputData, [1, 3, 320, 320]);

    const outputs = await session.run({
        data: tensor,
    });

    const output = outputs["output"];

    const anchors = generateGridAnchors(320, 320);

    const detections = postProcess(output.cpuData, anchors, 320, 320);

    const results = nms(detections);

    console.log(results);

    drawDetections(ctx, detections, classNames);
});
