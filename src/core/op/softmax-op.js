
import { Tensor } from "../tensor.js"
import { SoftmaxBackward } from "../autograd/softmax-backward.js"

export class SoftmaxOp {
    apply(input, dim = -1) {
        if (!(input instanceof Tensor)) {
            throw new Error("Must be tensor")
        }

        const rank = input.shape.length

        if (rank === 0) {
            throw new Error("Softmax requires a non-scalar tensor")
        }

        if (!Number.isInteger(dim)) {
            throw new Error("dim must be an integer")
        }

        if (dim < 0) {
            dim += rank
        }

        if (dim < 0 || dim >= rank) {
            throw new Error("Invalid softmax dimension")
        }

        const shape = input.shape
        const axisSize = shape[dim]
        const outputData = new Float32Array(input.numel)

        // Number of groups before and after the axis.
        const outerSize = shape
            .slice(0, dim)
            .reduce((a, b) => a * b, 1)

        const innerSize = shape
            .slice(dim + 1)
            .reduce((a, b) => a * b, 1)

        for (let outer = 0; outer < outerSize; outer++) {
            for (let inner = 0; inner < innerSize; inner++) {
                let max = -Infinity

                // Find the maximum for numerical stability.
                for (let k = 0; k < axisSize; k++) {
                    const index =
                        outer * axisSize * innerSize +
                        k * innerSize +
                        inner

                    const value = input.data[
                        this.#physicalIndex(input, index)
                    ]

                    if (value > max) {
                        max = value
                    }
                }

                let sum = 0

                // Compute exponentials and their sum.
                for (let k = 0; k < axisSize; k++) {
                    const index =
                        outer * axisSize * innerSize +
                        k * innerSize +
                        inner

                    const value = Math.exp(
                        input.data[this.#physicalIndex(input, index)] - max
                    )

                    outputData[index] = value
                    sum += value
                }

                // Normalize.
                for (let k = 0; k < axisSize; k++) {
                    const index =
                        outer * axisSize * innerSize +
                        k * innerSize +
                        inner

                    outputData[index] /= sum
                }
            }
        }

        const output = new Tensor(outputData, [...shape])

        if (input.requiresGrad) {
            output.requiresGrad = true
            output.gradFn = new SoftmaxBackward(
                [input],
                output,
                dim
            )
        }

        return output
    }

    #physicalIndex(tensor, logicalIndex) {
        let physicalIndex = tensor.offset

        for (let d = tensor.shape.length - 1; d >= 0; d--) {
            const coordinate = logicalIndex % tensor.shape[d]
            logicalIndex = Math.floor(logicalIndex / tensor.shape[d])

            physicalIndex += coordinate * tensor.strides[d]
        }

        return physicalIndex
    }
}
