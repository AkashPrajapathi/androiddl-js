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

    if (Array.isArray(data[0])) {
        let pred = data.every(v => Array.isArray(v) && v.length === data[0].length)

        if (pred) {
            for (let v of data) {
                validateShape(v)
            }
        } else {
            throw new Error("Invalid shape of tensor")
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

    if (data instanceof Float32Array) {
        return data
    }

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