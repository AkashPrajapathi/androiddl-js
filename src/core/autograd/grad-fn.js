export class GradFn {
    constructor(inputs) {
        if (!Array.isArray(inputs)) {
            throw new Error("inputs must be 1D array of tensors")
        }
        this.inputs = inputs
    }

    backward(outputGrad) {
        throw new Error("backward() must be implemented")
    }
}