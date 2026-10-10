import { normalizeShape, flattenToFloat32 } from "./util.js"
import { TensorIterator } from "../iterator/tensor-iterator.js"
import { AddOp } from '../op/add_op.js'
import { SubOp } from '../op/sub_op.js'
import { MulOp } from '../op/mul_op.js'
import { DivOp } from '../op/div_op.js'

export class Tensor {
    constructor(data, shape = []) {
        this.data = flattenToFloat32(data)

        if (shape.length === 0) {
            normalizeShape(data, shape)
        } else {
            this.#validateShape(data, shape)
        }

        this.shape = shape
        this.strides = this.#calculateStrides(shape)
        this.offset = 0
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


        const strides = new Array(shape.length)
        strides[shape.length - 1] = 1

        let s = 1

        for (let i = shape.length - 2; i >= 0; i--) {
            s *= shape[i + 1]
            strides[i] = s
        }

        return strides
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

        return this.permute(...indices)
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
            console.log("C-View")
            let x = new Tensor(this.data, shape)
            x.strides = this.#calculateStrides(shape)
            x.offset = this.offset
            return x
        }

        const newStrides = this.#calculateReshapeStrides(this.shape, this.strides, shape);

        if (newStrides !== null) {
            console.log("NC-View")
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

        console.log("Copy")
        let x = new Tensor(copiedData, shape)
        x.strides = this.#calculateStrides(shape)
        x.offset = this.offset
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

        // shape=[2,3,4]-> strides=[6,4,1]
        // strides[i] == shape[i+1]*strides[i+1]

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

        // [6, 4] -> [2,1]

        // 3,1,2,2,2

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

            // 2,3,2 -> 12 -> 2
            // 12, 4 ,2
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