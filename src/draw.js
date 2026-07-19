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

export function drawDetections(
    ctx,
    detections,
    classNames,
    scaleX = 1,
    scaleY = 1,
) {
    ctx.lineWidth = 2;
    ctx.font = "16px sans-serif";
    ctx.textBaseline = "top";

    for (const det of detections) {
        const { x1, y1, x2, y2, score, classId } = det;

        const color = COLORS[classId % COLORS.length];

        // 表示座標へ変換
        const sx1 = x1 * scaleX;
        const sy1 = y1 * scaleY;
        const sx2 = x2 * scaleX;
        const sy2 = y2 * scaleY;

        // Bounding Box
        ctx.strokeStyle = color;
        ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);

        // ラベル
        const className = classNames[classId];
        const label = `${className} ${(score * 100).toFixed(1)}%`;

        const textWidth = ctx.measureText(label).width;
        const textHeight = 18;

        let textX = sx1;
        let textY = sy1 - textHeight;

        // 上にはみ出る場合はボックス内に表示
        if (textY < 0) {
            textY = sy1;
        }

        // ラベル背景
        ctx.fillStyle = color;
        ctx.fillRect(textX, textY, textWidth + 8, textHeight);

        // ラベル文字
        ctx.fillStyle = "white";
        ctx.fillText(label, textX + 4, textY + 2);
    }
}
