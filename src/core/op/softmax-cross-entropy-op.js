import { Tensor } from "../tensor.js";
import { SoftmaxCrossEntropyBackward } from "../autograd/softmax-cross-entropy-backward.js";

export class SoftmaxCrossEntropyOp {
    apply(logits, target, dim = -1, reduction = "mean") {
        if (!(logits instanceof Tensor) || !(target instanceof Tensor)) {
            throw new Error("Logits and target must be Tensors");
        }

        if (!["none", "sum", "mean"].includes(reduction)) {
            throw new Error(`Invalid reduction: ${reduction}`);
        }

        const rank = logits.shape.length;

        if (rank === 0) {
            throw new Error("Logits must have at least one dimension");
        }

        if (!Number.isInteger(dim)) {
            throw new Error("dim must be an integer");
        }

        if (dim < 0) dim += rank;

        if (dim < 0 || dim >= rank) {
            throw new Error("Invalid class dimension");
        }

        const shape = logits.shape;
        const classes = shape[dim];

        const targetShape = shape.filter((_, i) => i !== dim);

        if (
            target.shape.length !== targetShape.length ||
            !target.shape.every((v, i) => v === targetShape[i])
        ) {
            throw new Error(
                `Target shape must be [${targetShape}], got [${target.shape}]`
            );
        }

        const outerSize = shape
            .slice(0, dim)
            .reduce((a, b) => a * b, 1);

        const innerSize = shape
            .slice(dim + 1)
            .reduce((a, b) => a * b, 1);

        const sampleCount = outerSize * innerSize;
        const targetValues = Array.from(target);

        // Validate targets before calculating the loss.
        for (const label of targetValues) {
            if (
                !Number.isInteger(label) ||
                label < 0 ||
                label >= classes
            ) {
                throw new Error(`Invalid class index: ${label}`);
            }
        }

        const losses = new Float32Array(sampleCount);
        const probabilities = new Float32Array(logits.numel);

        for (let outer = 0; outer < outerSize; outer++) {
            for (let inner = 0; inner < innerSize; inner++) {
                const sample = outer * innerSize + inner;
                const label = targetValues[sample];

                let max = -Infinity;

                // Find maximum logit.
                for (let k = 0; k < classes; k++) {
                    const logicalIndex =
                        outer * classes * innerSize +
                        k * innerSize +
                        inner;

                    const value = logits.data[
                        this.#physicalIndex(logits, logicalIndex)
                    ];

                    if (!Number.isFinite(value)) {
                        throw new Error(
                            "Logits must contain finite values"
                        );
                    }

                    if (value > max) max = value;
                }

                // Stable log-sum-exp and softmax probabilities.
                let sumExp = 0;

                for (let k = 0; k < classes; k++) {
                    const logicalIndex =
                        outer * classes * innerSize +
                        k * innerSize +
                        inner;

                    const physicalIndex = this.#physicalIndex(
                        logits,
                        logicalIndex
                    );

                    const probability = Math.exp(
                        logits.data[physicalIndex] - max
                    );

                    probabilities[logicalIndex] = probability;
                    sumExp += probability;
                }

                for (let k = 0; k < classes; k++) {
                    const logicalIndex =
                        outer * classes * innerSize +
                        k * innerSize +
                        inner;

                    probabilities[logicalIndex] /= sumExp;
                }

                const targetLogicalIndex =
                    outer * classes * innerSize +
                    label * innerSize +
                    inner;

                // log(sum(exp(logits - max))) + max - targetLogit
                const targetLogit = logits.data[
                    this.#physicalIndex(logits, targetLogicalIndex)
                ];

                losses[sample] =
                    max + Math.log(sumExp) - targetLogit;
            }
        }

        let output;
        let scale = 1;

        if (reduction === "none") {
            output = new Tensor(losses, targetShape);
        } else {
            let total = 0;

            for (const loss of losses) total += loss;

            if (reduction === "mean") {
                total /= sampleCount;
                scale = 1 / sampleCount;
            }

            output = new Tensor(new Float32Array([total]), []);
        }

        if (logits.requiresGrad) {
            output.requiresGrad = true;
            output.gradFn = new SoftmaxCrossEntropyBackward(
                [logits],
                probabilities,
                targetValues,
                shape,
                targetShape,
                dim,
                reduction,
                scale
            );
        }

        return output;
    }

    #physicalIndex(tensor, logicalIndex) {
        let physicalIndex = tensor.offset;

        for (let d = tensor.shape.length - 1; d >= 0; d--) {
            const coordinate = logicalIndex % tensor.shape[d];
            logicalIndex = Math.floor(logicalIndex / tensor.shape[d]);

            physicalIndex += coordinate * tensor.strides[d];
        }

        return physicalIndex;
    }
}
