import { TensorIterator } from '../iterator/tensor-iterator.js'
import { Tensor } from './tensor.js'

function validateShape(data) {

    if (data instanceof Float32Array) {
        return
    }

    if (!Array.isArray(data)) {
        return
    }

    if (data.length === 0) {
        return
    }

    const isFirstArray = Array.isArray(data[0])

    const allMatch = data.every(v => Array.isArray(v) === isFirstArray)

    if (!allMatch) {
        throw new Error('Invalid data shape: mixed arrays and scalars at the same level');
    }


    if (isFirstArray) {
        // Ensure all sub-arrays have the exact same length
        let targetLength = data[0].length;
        let validLengths = data.every(v => v.length === targetLength);

        if (!validLengths) {
            throw new Error('Invalid data shape: jagged sub-array lengths');
        }

        // Recurse into each sub-array
        for (let e of data) {
            validateShape(e);
        }
    } else {
        // We are at the leaf level. Ensure every element is a real number!
        let allNumbers = data.every(v => typeof v === 'number' && !Number.isNan ? !Number.isNaN(v) : !isNaN(v));

        if (!allNumbers) {
            throw new Error('Invalid data: All tensor elements must be numbers');
        }
    }
}

export function normalizeShape(data, shape) {
    validateShape(data)

    function getShape(data, shape) {

        if (!Array.isArray(data)) {
            return
        }

        if (data.length === 0) {
            return
        }

        shape.push(data.length)

        if (Array.isArray(data[0])) {
            getShape(data[0], shape)
        }
    }

    if (data instanceof Float32Array) {
        shape.push(data.length)
        return
    }

    getShape(data, shape)
}

function countElements(data) {

    const stack = [data]

    let count = 0

    while (stack.length > 0) {
        let current = stack.pop()

        if (Array.isArray(current)) {
            for (let i = current.length - 1; i >= 0; i--) {
                stack.push(current[i])
            }
        } else if (typeof current === "number") {
            count++
        }
    }

    return count
}

export function flattenToFloat32(data) {

    if (data instanceof Float32Array) {
        return data
    }

    const totalElements = countElements(data)

    const resultView = new Float32Array(totalElements)

    const stack = [data]

    let count = 0

    while (stack.length > 0) {
        let current = stack.pop()

        if (Array.isArray(current)) {
            for (let i = current.length - 1; i >= 0; i--) {
                stack.push(current[i])
            }
        } else if (typeof current === "number") {
            resultView[count++] = current
        }
    }

    return resultView
}

// aShape = [2,4,3]
// bShape = [3]
// result = [2,4,3]

// aDim===bDim || aDim === 1 || bDim === 1

export function broadcastShape(aShape, bShape) {
    if (!Array.isArray(aShape) || !Array.isArray(bShape)) {
        throw new Error("Shape must be 1D array")
    }

    let isAInt = aShape.every(v => Number.isInteger(v) && v > 0);
    let isBInt = bShape.every(v => Number.isInteger(v) && v > 0);

    if (!isAInt || !isBInt) {
        throw new Error("Shape must contain positive integer")
    }

    const resultRank = Math.max(aShape.length, bShape.length)

    const broadcastedShape = new Array(resultRank)

    for (let i = 0; i < resultRank; i++) {
        const aIdx = aShape.length - 1 - (resultRank - 1 - i)
        const bIdx = bShape.length - 1 - (resultRank - 1 - i)

        const aDim = aIdx >= 0 ? aShape[aIdx] : 1
        const bDim = bIdx >= 0 ? bShape[bIdx] : 1

        if (aDim === bDim || aDim === 1 || bDim === 1) {
            broadcastedShape[i] = Math.max(aDim, bDim)
        } else {
            throw new Error(`Cannot broadcast shape ${aShape.toString()} and ${bShape.toString()}`)
        }
    }

    return broadcastedShape

}

// aShape = [2,1,3] -> [3,3,1] -> [3,3,1]
// bShape = [3] -> [1] -> [0,0,1]
// result = [2,1,3] -> [3,3,1]


export function broadcastStrides(inputShape, inputStrides, outputShape) {
    const rankDiff = outputShape.length - inputShape.length

    const result = new Array(outputShape.length)

    for (let i = 0; i < outputShape.length; i++) {
        const inputDim = i - rankDiff

        if (inputDim < 0) {
            // Missing leading dimension
            result[i] = 0
        } else if (
            inputShape[inputDim] === 1 &&
            outputShape[i] !== 1
        ) {
            // Broadcast a singleton dimension
            result[i] = 0
        } else {
            // Preserve the corresponding input stride
            result[i] = inputStrides[inputDim]
        }
    }

    return result
}

export function broadcastIterator(tensor, outputShape) {

    if (!(tensor instanceof Tensor)) {
        throw new Error("Expects tensor object")
    }

    const strides = broadcastStrides(tensor.shape, tensor.strides, outputShape)

    return new TensorIterator(
        tensor.data,
        outputShape,
        strides,
        tensor.offset
    )

}

export function elementWiseOp(a, b, operator) {

    if (!(a instanceof Tensor) || !(b instanceof Tensor)) {
        throw new Error("a and b must be tensor objects")
    }

    const outputShape = broadcastShape(a.shape, b.shape)

    const result = new Float32Array(
        outputShape.reduce((acc, v) => acc * v, 1)
    )

    const aIterator = broadcastIterator(a, outputShape)
    const bIterator = broadcastIterator(b, outputShape)

    let i = 0

    while (aIterator.hasNext()) {
        result[i++] = operator(aIterator.next(), bIterator.next())
    }

    return new Tensor(result, outputShape)
}