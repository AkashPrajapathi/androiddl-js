import { TensorIterator } from './iterator/tensor-iterator.js'
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

export function outputIndex(coordinates, shape) {
    let index = 0;
    let stride = 1;

    for (let d = shape.length - 1; d >= 0; d--) {
        index += coordinates[d] * stride;
        stride *= shape[d];
    }

    return index;
}

export function computeReducedShape(shape, axes, keepdim) {
    if (keepdim) {
        const output = [...shape]
        for (const axis of axes) {
            output[axis] = 1
        }

        return output
    }

    const output = new Array(shape.length - axes.length).fill(0)

    let outputIndex = 0

    for (let i = 0; i < shape.length; i++) {
        if (axes.includes(i)) {
            continue
        }

        output[outputIndex++] = shape[i]
    }
    return output

}

export function sumToShape(outputGrad, targetShape) {
    if (!(outputGrad instanceof Tensor)) {
        throw new Error("outputGrad must be tensor object");
    }

    if (!Array.isArray(targetShape)) {
        throw new Error("targetShape must be an array");
    }

    // Scalar source → broadcast to target shape.
    if (outputGrad.shape.length === 0) {
        const size = targetShape.reduce(
            (acc, dim) => acc * dim,
            1
        );

        return new Tensor(
            new Float32Array(size).fill(
                outputGrad.data[outputGrad.offset]
            ),
            targetShape
        );
    }

    if (targetShape.length > outputGrad.shape.length) {
        throw new Error(
            `Cannot reduce shape ${outputGrad.shape} to ${targetShape}`
        );
    }

    const leadingDimensions =
        outputGrad.shape.length - targetShape.length;

    // Validate compatible dimensions.
    for (let i = 0; i < targetShape.length; i++) {
        const sourceDim =
            outputGrad.shape[leadingDimensions + i];

        const targetDim = targetShape[i];

        if (targetDim !== sourceDim && targetDim !== 1) {
            throw new Error(
                `Cannot reduce shape ${outputGrad.shape} to ${targetShape}`
            );
        }
    }

    const axes = [];

    // Reduce extra leading dimensions.
    for (let i = 0; i < leadingDimensions; i++) {
        axes.push(i);
    }

    // Reduce dimensions that were broadcast from size 1.
    for (let i = 0; i < targetShape.length; i++) {
        const outputAxis = leadingDimensions + i;

        if (
            targetShape[i] === 1 &&
            outputGrad.shape[outputAxis] !== 1
        ) {
            axes.push(outputAxis);
        }
    }

    if (axes.length === 0) {
        return outputGrad;
    }

    const reduced = outputGrad.sum(axes, true);

    return reduced.reshape(targetShape);
}

export function expandReducedAxes(input, outputShape) {
    if (!(input instanceof Tensor)) {
        throw new Error("expandReducedAxes: input must be a tensor")
    }

    if (!Array.isArray(outputShape)) {
        throw new Error("expandReducedAxes: outputShape must be 1D array")
    }

    const size = outputShape.reduce((a, b) => a * b, 1)

    const iter = broadcastIterator(input, outputShape)
    const result = new Float32Array(size)

    let i = 0

    while (iter.hasNext()) {
        result[i++] = iter.next()
    }

    return new Tensor(result, [...outputShape])
}

export function inversePermutation(dimensions) {
    if (!Array.isArray()) {
        throw new Error("dimensions must be 1D array")
    }

    if (!dimensions.every(v => Number.isInteger(v) && v >= 0)) {
        throw new Error("dimensions must have valid axes")
    }

    const inverse = new Array(dimensions.length)

    for (const i in dimensions) {
        inverse[dimensions[i]] = i
    }

    return inverse
}

export function onesLike(tensor) {
    const data = new Float32Array(tensor.numel).fill(1)

    return new Tensor(data, [...tensor.shape])
}