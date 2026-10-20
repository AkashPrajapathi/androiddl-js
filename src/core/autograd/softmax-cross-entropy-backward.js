import { Tensor } from "../tensor.js";
import { GradFn } from "./grad-fn.js";

export class SoftmaxCrossEntropyBackward extends GradFn {
    constructor(
        inputs,
        probabilities,
        targets,
        shape,
        targetShape,
        dim,
        reduction,
        scale
    ) {
        super(inputs);

        this.probabilities = probabilities;
        this.targets = targets;
        this.shape = shape;
        this.targetShape = targetShape;
        this.dim = dim;
        this.reduction = reduction;
        this.scale = scale;
    }

    backward(outputGrad) {
        const shape = this.shape;
        const classes = shape[this.dim];

        const outerSize = shape
            .slice(0, this.dim)
            .reduce((a, b) => a * b, 1);

        const innerSize = shape
            .slice(this.dim + 1)
            .reduce((a, b) => a * b, 1);

        const sampleCount = outerSize * innerSize;
        const gradData = new Float32Array(this.inputs[0].numel);

        let sample = 0;

        for (let outer = 0; outer < outerSize; outer++) {
            for (let inner = 0; inner < innerSize; inner++) {
                let upstream = this.scale;

                if (this.reduction === "none") {
                    upstream = outputGrad.data[
                        this.#physicalIndex(outputGrad, sample)
                    ];
                } else if (this.reduction === "sum") {
                    upstream = outputGrad.data[
                        this.#physicalIndex(outputGrad, 0)
                    ];
                } else {
                    upstream *= outputGrad.data[
                        this.#physicalIndex(outputGrad, 0)
                    ];
                }

                const label = this.targets[sample];

                for (let k = 0; k < classes; k++) {
                    const index =
                        outer * classes * innerSize +
                        k * innerSize +
                        inner;

                    let grad = this.probabilities[index];

                    if (k === label) grad -= 1;

                    gradData[index] = grad * upstream;
                }

                sample++;
            }
        }

        return [new Tensor(gradData, shape)];
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
