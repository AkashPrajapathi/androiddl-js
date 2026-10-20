import { Tensor } from "../tensor.js"
import { GradFn } from "./grad-fn.js"

export class SoftmaxBackward extends GradFn {
    constructor(inputs, output, dim) {
        this.inputs = inputs
        this.output = output
        this.dim = dim
    }

    backward(outGrad) {
        const output = this.output
        const shape = output.shape
        const dim = this.dim

        const axisSize = shape[dim]

        const outerSize = shape
            .slice(0, dim)
            .reduce((a, b) => a * b, 1)

        const innerSize = shape
            .slice(dim + 1)
            .reduce((a, b) => a * b, 1)

        const gradData = new Float32Array(output.numel)

        for (let outer = 0; outer < outerSize; outer++) {
            for (let inner = 0; inner < innerSize; inner++) {
                let dot = 0

                // dot = sum(outGrad * output, axis)
                for (let k = 0; k < axisSize; k++) {
                    const index =
                        outer * axisSize * innerSize +
                        k * innerSize +
                        inner

                    const y = this.#valueAt(output, index)
                    const g = this.#valueAt(outGrad, index)

                    dot += y * g
                }

                // inGrad = output * (outGrad - dot)
                for (let k = 0; k < axisSize; k++) {
                    const index =
                        outer * axisSize * innerSize +
                        k * innerSize +
                        inner

                    const y = this.#valueAt(output, index)
                    const g = this.#valueAt(outGrad, index)

                    gradData[index] = y * (g - dot)
                }
            }
        }

        return [new Tensor(gradData, [...shape])]
    }

    #valueAt(tensor, logicalIndex) {
        let physicalIndex = tensor.offset

        for (let d = tensor.shape.length - 1; d >= 0; d--) {
            const coordinate = logicalIndex % tensor.shape[d]
            logicalIndex = Math.floor(logicalIndex / tensor.shape[d])

            physicalIndex += coordinate * tensor.strides[d]
        }

        return tensor.data[physicalIndex]
    }
}
