const COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
];

export function drawDetections(ctx, detections, classNames) {
    ctx.lineWidth = 2;
    ctx.font = "16px sans-serif";
    ctx.textBaseline = "top";

    for (const det of detections) {
        const { x1, y1, x2, y2, score, classId } = det;

        const color = COLORS[classId % COLORS.length];

        // Bounding Box
        ctx.strokeStyle = color;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // ラベル
        const className = classNames[classId];

        const label = `${className} ${(score * 100).toFixed(1)}%`;

        const textWidth = ctx.measureText(label).width;
        const textHeight = 18;

        let textX = x1;
        let textY = y1 - textHeight;

        // 上にはみ出る場合はボックス内に表示
        if (textY < 0) {
            textY = y1;
        }

        // ラベル背景
        ctx.fillStyle = color;
        ctx.fillRect(textX, textY, textWidth + 8, textHeight);

        // ラベル文字
        ctx.fillStyle = "white";
        ctx.fillText(label, textX + 4, textY + 2);
    }
}
