import { ConcatenateBackward } from "../autograd/concatenate-backward.js"
import { TensorIterator } from "../iterator/tensor-iterator.js"
import { Tensor } from "../tensor.js"

export class ConcatenateOp {
    apply(tensors, dim) {
        if (!Array.isArray(tensors) || tensors.length === 0) {
            throw new Error("Concatenate requires at least one tensor")
        }

        for (const tensor of tensors) {
            if (!(tensor instanceof Tensor)) {
                throw new Error("All must be tensor object")
            }
        }

        const rank = tensors[0].shape.length

        // Validate rank
        for (const tensor of tensors) {
            if (tensor.shape.length !== rank) {
                throw new Error("All tensors must have the same rank")
            }
        }

        // Normalize negative dimension
        if (dim < 0) {
            dim += rank
        }

        if (dim < 0 || dim >= rank) {
            throw new Error("Invalid concatenation dimension")
        }

        // Validate non-concatenation dimensions
        for (let i = 1; i < tensors.length; i++) {
            for (let d = 0; d < rank; d++) {
                if (d === dim) continue

                if (tensors[i].shape[d] !== tensors[0].shape[d]) {
                    throw new Error(`Shape mismatch at dimension ${d}`)
                }
            }
        }

        const outputShape = [...tensors[0].shape]

        outputShape[dim] = tensors.reduce(
            (sum, tensor) => sum + tensor.shape[dim], 0)

        const outputSize = outputShape.reduce((a, v) => a * v, 1)

        const outputData = new Float32Array(outputSize)
        const output = new Tensor(outputData, outputShape)

        let offset = 0

        for (const tensor of tensors) {

            const iter = new TensorIterator(
                tensor.data,
                tensor.shape,
                tensor.strides,
                tensor.offset
            )

            while (iter.hasNext()) {

                const coordinate = iter.getCoordinate()
                const value = iter.next()

                const outputCoord = [...coordinate]
                outputCoord[dim] += offset

                const outputIndex = outputCoord.reduce(
                    (acc, v, i) => acc + v * output.strides[i],
                    output.offset
                )

                output.data[outputIndex] = value
            }

            offset += tensor.shape[dim]
        }

        if (tensors.some(v => v.requiresGrad)) {
            output.requiresGrad = true
            output.gradFn = new ConcatenateBackward([...tensors], dim)
        }

        return output
    }
}