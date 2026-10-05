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