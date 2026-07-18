export function generateGridAnchors(inputHeight, inputWidth) {
    const strideValues = [8, 16, 32, 64];

    let totalCells = 0;
    for (const stride of strideValues) {
        const featH = Math.ceil(inputHeight / stride);
        const featW = Math.ceil(inputWidth / stride);
        totalCells += featH * featW;
    }

    const cx = new Float32Array(totalCells);
    const cy = new Float32Array(totalCells);
    const strides = new Float32Array(totalCells);

    let index = 0;

    for (const stride of strideValues) {
        const featH = Math.ceil(inputHeight / stride);
        const featW = Math.ceil(inputWidth / stride);

        for (let y = 0; y < featH; y++) {
            for (let x = 0; x < featW; x++) {
                cx[index] = x + 0.5;
                cy[index] = y + 0.5;
                strides[index] = stride;
                index++;
            }
        }
    }

    return { cx, cy, strides };
}

function softmax8(input, offset, output) {
    let max = input[offset];

    for (let i = 1; i < 8; i++) {
        if (input[offset + i] > max) {
            max = input[offset + i];
        }
    }

    let sum = 0;

    for (let i = 0; i < 8; i++) {
        const value = Math.exp(input[offset + i] - max);
        output[i] = value;
        sum += value;
    }

    for (let i = 0; i < 8; i++) {
        output[i] /= sum;
    }
}

const probs = new Float32Array(8);

function decodeDistance(input, offset) {
    softmax8(input, offset, probs);

    let distance = 0;

    for (let i = 0; i < 8; i++) {
        distance += probs[i] * i;
    }

    return distance;
}

export function postProcess(
    output,
    anchors,
    inputWidth,
    inputHeight,
    scoreThreshold = 0.35,
) {
    const detections = [];

    const { cx, cy, strides } = anchors;
    const numAnchors = cx.length;

    for (let i = 0; i < numAnchors; i++) {
        const base = i * 112;

        // 最大クラス探索
        let classId = 0;
        let score = output[base];

        for (let c = 1; c < 80; c++) {
            const s = output[base + c];
            if (s > score) {
                score = s;
                classId = c;
            }
        }

        if (score < scoreThreshold) {
            continue;
        }

        // DFL Decode
        const left = decodeDistance(output, base + 80);
        const top = decodeDistance(output, base + 88);
        const right = decodeDistance(output, base + 96);
        const bottom = decodeDistance(output, base + 104);

        const stride = strides[i];

        let x1 = (cx[i] - left) * stride;
        let y1 = (cy[i] - top) * stride;
        let x2 = (cx[i] + right) * stride;
        let y2 = (cy[i] + bottom) * stride;

        x1 = Math.max(0, Math.min(inputWidth, x1));
        y1 = Math.max(0, Math.min(inputHeight, y1));
        x2 = Math.max(0, Math.min(inputWidth, x2));
        y2 = Math.max(0, Math.min(inputHeight, y2));

        detections.push({
            x1,
            y1,
            x2,
            y2,
            score,
            classId,
        });
    }

    return detections;
}

function iou(a, b) {
    const x1 = Math.max(a.x1, b.x1);
    const y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2);
    const y2 = Math.min(a.y2, b.y2);

    const w = Math.max(0, x2 - x1);
    const h = Math.max(0, y2 - y1);

    const intersection = w * h;

    const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);

    const union = areaA + areaB - intersection;

    return union > 0 ? intersection / union : 0;
}

export function nms(detections, iouThreshold = 0.45) {
    const sorted = [...detections].sort((a, b) => b.score - a.score);

    const result = [];

    while (sorted.length > 0) {
        const best = sorted.shift();
        result.push(best);

        for (let i = sorted.length - 1; i >= 0; i--) {
            if (sorted[i].classId !== best.classId) {
                continue;
            }

            if (iou(best, sorted[i]) >= iouThreshold) {
                sorted.splice(i, 1);
            }
        }
    }

    return result;
}
