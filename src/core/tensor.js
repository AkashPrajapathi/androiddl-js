import { normalizeShape, flattenToFloat32 } from "./util.js"
import { TensorIterator } from "../iterator/tensor-iteraror.js"

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