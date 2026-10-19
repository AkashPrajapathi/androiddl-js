import { normalizeShape, flattenToFloat32, onesLike } from "./util.js"
import { TensorIterator } from "./iterator/tensor-iterator.js"
import { AddOp } from './op/add-op.js'
import { SubOp } from './op/sub-op.js'
import { MulOp } from './op/mul-op.js'
import { DivOp } from './op/div-op.js'
import { SumOp } from "./op/sum-op.js"
import { MinOp } from "./op/min-op.js"
import { MaxOp } from "./op/max-op.js"
import { MeanOp } from "./op/mean-op.js"
import { MatMulOp } from "./op/matmul-op.js"
import { PermuteBackward } from "./autograd/permute-backward.js"
import { TransposeBackward } from "./autograd/transpose-backward.js"
import { ReshapeBackward } from "./autograd/reshape-backward.js"
import { ReLUOp } from "./op/relu-op.js"

export class Tensor {
    constructor(data, shape, requiresGrad = false) {
        this.data = flattenToFloat32(data);

        if (shape === undefined) {
            shape = [];
            normalizeShape(data, shape);
        } else {
            this.#validateShape(data, shape);
        }

        this.shape = shape;
        this.strides = this.#calculateStrides(shape);
        this.offset = 0;

        this.numel = this.data.length

        this.requiresGrad = requiresGrad
        this.grad = null
        this.gradFn = null
    }

    #validateShape(data, shape) {

        if (!Array.isArray(shape)) {
            throw new Error("Shape must be 1D array")
        }

        for (let v of shape) {
            if (!Number.isInteger(v) || v <= 0) {
                throw new Error("Shape must contain positive integers")
            }
        }

        const currentShape = []

        normalizeShape(data, currentShape)

        const oldSize = currentShape.reduce((acc, v) => acc * v, 1)
        const newSize = shape.reduce((acc, v) => acc * v, 1)

        if (oldSize !== newSize) {
            throw new Error(`Invalid reshape: ${oldSize} cannot become ${newSize}`)
        }

    }

    #calculateStrides(shape) {
        if (shape.length === 0) {
            return [];
        }

        const strides = new Array(shape.length);
        strides[shape.length - 1] = 1;

        let s = 1;

        for (let i = shape.length - 2; i >= 0; i--) {
            s *= shape[i + 1];
            strides[i] = s;
        }

        return strides;
    }

    [Symbol.iterator]() {
        const iter = new TensorIterator(this.data, this.shape, this.strides, this.offset)

        return {
            next() {
                if (iter.hasNext()) {
                    return { value: iter.next(), done: false }
                }

                return { value: undefined, done: true }
            }
        }
    }

    permute(...indices) {
        if (indices.length !== this.shape.length) {
            throw new Error(`Indices must contain ${this.shape.length}`)
        }

        const set = new Set(indices)

        if (set.size < indices.length) {
            throw new Error("Indices can not have duplicate value")
        }

        let isValidRange = true;

        for (let i = 0; i < indices.length; i++) {
            if (indices[i] < 0 || indices[i] >= this.shape.length) {
                isValidRange = false
                break
            }
        }

        if (!isValidRange) {
            throw new Error("OutOfIndex value")
        }

        const newShape = new Array(this.shape.length)
        const newStrides = new Array(this.shape.length)

        for (let i = 0; i < this.shape.length; i++) {
            newShape[i] = this.shape[indices[i]]
            newStrides[i] = this.strides[indices[i]]
        }

        const x = new Tensor(this.data, newShape)
        x.strides = newStrides
        x.offset = this.offset

        if (this.requiresGrad) {
            x.requiresGrad = true
            x.gradFn = new PermuteBackward([this], [...indices])
        }

        return x
    }


    transpose(dim0, dim1) {
        if (dim0 >= dim1) {
            throw new Error("dim0 can not greater or eqault to dim1")
        }

        const indices = new Array(this.shape.length)
        for (let i = 0; i < this.shape.length; i++) {
            indices[i] = i
        }

        let a = indices[dim0]
        let b = indices[dim1]

        indices[dim0] = b;
        indices[dim1] = a;

        const output = this.permute(...indices)

        if (this.requiresGrad) {
            output.requiresGrad = true
            output.gradFn = new TransposeBackward([this], dim0, dim1)
        }

        return output
    }

    reshape(shape) {
        if (!Array.isArray(shape)) {
            throw new Error("Shape must be 1D array")
        }

        let pred = shape.every((v) => Number.isInteger(v) && v >= 0)
        if (!pred) {
            throw new Error("Shape must contain positive integer")
        }

        const count = shape.reduce((acc, v) => acc * v, 1);
        if (count !== this.data.length) {
            throw new Error(`Invalid reshape: ${count} cannot become ${this.data.length}`)
        }

        if (this.#isContiguous(this.shape, this.strides)) {
            let x = new Tensor(this.data, shape)
            x.strides = this.#calculateStrides(shape)
            x.offset = this.offset
            return x
        }

        const newStrides = this.#calculateReshapeStrides(this.shape, this.strides, shape);

        if (newStrides !== null) {
            let x = new Tensor(this.data, shape)
            x.strides = newStrides
            x.offset = this.offset
            return x
        }

        // Copy reshape
        const copiedData = new Float32Array(count)

        let i = 0

        for (let v of this) {
            copiedData[i] = v
            i++
        }

        let x = new Tensor(copiedData, shape)
        x.strides = this.#calculateStrides(shape)
        x.offset = this.offset


        if (this.requiresGrad) {
            x.requiresGrad = true
            x.gradFn = new ReshapeBackward([this])
        }

        return x

    }

    add(other) {
        if (!(other instanceof Tensor)) {
            throw new Error("Must be tensor")
        }

        let op = new AddOp()

        return op.apply(this, other)
    }

    sub(other) {
        if (!(other instanceof Tensor)) {
            throw new Error("Must be tensor")
        }

        let op = new SubOp()

        return op.apply(this, other)
    }

    mul(other) {
        if (!(other instanceof Tensor)) {
            throw new Error("Must be tensor")
        }

        let op = new MulOp()

        return op.apply(this, other)
    }

    div(other) {
        if (!(other instanceof Tensor)) {
            throw new Error("Must be tensor")
        }

        let op = new DivOp()

        return op.apply(this, other)
    }

    sum(axes, keepdim = false) {
        const op = new SumOp()

        return op.apply(this, axes, keepdim)
    }

    min(axes, keepdim = false) {
        const op = new MinOp()

        return op.apply(this, axes, keepdim)
    }

    max(axes, keepdim = false) {
        const op = new MaxOp()

        return op.apply(this, axes, keepdim)
    }

    mean(axes, keepdim = false) {
        const op = new MeanOp()

        return op.apply(this, axes, keepdim)
    }

    matmul(other) {
        if (!(other instanceof Tensor)) {
            throw new Error("Must be tensor")
        }

        const op = new MatMulOp()

        return op.apply(this, other)
    }

    relu() {
        const op = new ReLUOp()
        return op.apply(this)
    }

    #isContiguous(shape, strides) {
        let expectedStride = 1

        for (let i = shape.length - 1; i >= 0; i--) {
            if (strides[i] !== expectedStride) {
                return false
            }

            expectedStride *= shape[i]
        }

        return true
    }

    #calculateReshapeStrides(shape, strides, newShape) {

        const size = shape.reduce((a, b) => a * b, 1)
        const newSize = newShape.reduce((a, b) => a * b, 1)

        if (size !== newSize) {
            return null
        }

        // Chunking
        const chunks = []
        let chunkSize = shape[shape.length - 1]
        let chunkStride = strides[strides.length - 1]

        for (let i = shape.length - 2; i >= 0; i--) {
            if (strides[i] == shape[i + 1] * strides[i + 1]) {
                chunkSize *= shape[i]
            } else {
                chunks.push({ size: chunkSize, stride: chunkStride })

                chunkSize = shape[i]
                chunkStride = strides[i]
            }
        }

        chunks.push({ size: chunkSize, strides, chunkStride })

        chunks.reverse()

        // Fitting
        const newStrides = new Array(newShape.length)
        let chunkIndex = 0
        let remainingChunkSize = chunks[0].size
        let currentChunkStride = chunks[0].stride

        for (let i = 0; i < newShape.length; i++) {
            const dim = newShape[i]

            if (dim === 1) {
                newStrides[i] = remainingChunkSize * currentChunkStride
                continue
            }

            if (remainingChunkSize % dim !== 0) {
                return null
            }

            remainingChunkSize /= dim
            newStrides[i] = remainingChunkSize * currentChunkStride

            if (remainingChunkSize === 1) {
                chunkIndex++

                if (chunkIndex < chunks.length) {
                    remainingChunkSize = chunks[chunkIndex].size
                    currentChunkStride = chunks[chunkIndex].stride
                }
            }
        }

        return (remainingChunkSize === 1 && chunkIndex === chunks.length) ? newStrides : null

    }

    backward(grad = null) {

        if (grad === null) {
            if (this.numel !== 1) {
                throw new Error('Gradient must be provided for multi-element tensors')
            }

            grad = onesLike(this)

        } else {
            if (!(grad instanceof Tensor)) {
                throw new Error("grad must be tensor of same shape")
            }

            for (let i = 0; i < this.shape; i++) {
                if (grad.shape[i] !== this.shape[i]) {
                    throw new Error(`grad of shape ${grad.shape.toString()} is not valid for tensor of shape ${this.shape.toString()}`)
                }
            }
        }


        this.#backward(grad)
    }

    #buildTopoSort(node) {
        const visited = new Set()
        const order = []

        function visit(node, visited, order) {
            if (visited.has(node)) {
                return
            }

            visited.add(node)

            const gradFn = node.gradFn

            if (gradFn !== null) {
                for (const v of gradFn.inputs) {
                    visit(v, visited, order)
                }
            }

            order.push(node)
        }

        visit(node, visited, order)

        return order.reverse()
    }

    #backward(grad) {
        const order = this.#buildTopoSort(this);

        // Seed the output gradient.
        this.grad = grad;

        for (const node of order) {
            if (node.gradFn === null) {
                continue;
            }

            const gradFn = node.gradFn;
            const inputGrads = gradFn.backward(node.grad);

            for (let i = 0; i < gradFn.inputs.length; i++) {
                const inputGrad = inputGrads[i];
                const input = gradFn.inputs[i];

                if (input.requiresGrad) {
                    if (input.grad === null) {
                        input.grad = inputGrad;
                    } else {
                        input.grad = input.grad.add(inputGrad);
                    }
                }
            }
        }
    }

    toString() {
        const iter = new TensorIterator(
            this.data,
            this.shape,
            this.strides,
            this.offset
        )

        if (this.shape.length === 0) {
            return "" + iter.next()
        }

        return `tensor(\n${this.#appendTensor(iter, 0)}\n\nshape   = [${this.shape.toString()}]\nstrides = [${this.strides.toString()}])`
    }

    #appendTensor(iter, dimension) {
        let builder = ""

        builder += "["

        const size = this.shape[dimension]

        for (let i = 0; i < size; i++) {

            if (dimension === this.shape.length - 1) {
                builder += `${iter.next()}`
            } else {
                builder += this.#appendTensor(iter, dimension + 1)
            }

            if (i < size - 1) {
                if (dimension === this.shape.length - 1) {
                    builder += " "
                } else {
                    builder += "\n"

                    for (let j = 0; j < dimension + 1; j++) {
                        builder += " "
                    }
                }
            }
        }

        builder += "]"

        return builder
    }
}