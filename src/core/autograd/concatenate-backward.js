import { TensorIterator } from "../iterator/tensor-iterator.js";
import { Tensor } from "../tensor.js";
import { GradFn } from "./grad-fn.js";

export class ConcatenateBackward extends GradFn {
    constructor(inputs, dim) {
        super(inputs)
        this.dim = dim
    }

    backward(outputGrad) {

        const grads = []

        let offset = 0

        for (const tensor of this.inputs) {

            const iter = new TensorIterator(
                tensor.data,
                tensor.shape,
                tensor.strides,
                tensor.offset
            )

            const data = new Float32Array(tensor.numel)
            const grad = new Tensor(data, [...tensor.shape])

            while (iter.hasNext()) {

                const coordinate = iter.getCoordinate()
                iter.next()

                const outputCoord = [...coordinate]
                outputCoord[this.dim] += offset

                const outputIndex = outputCoord.reduce(
                    (acc, v, i) => acc + v * outputGrad.strides[i],
                    outputGrad.offset
                )
                const gradIndex = coordinate.reduce(
                    (acc, v, i) => acc + v * grad.strides[i],
                    grad.offset
                )

                grad.data[gradIndex] = outputGrad.data[outputIndex]
            }

            offset += tensor.shape[this.dim]
            grads.push(grad)
        }

        return grads
    }
}