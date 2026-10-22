export class Optimizer {
    constructor(params, defaults) {
        this.params = [...params]
        this.defaults = defaults
        this.state = new Map()
    }

    zeroGrad(setToNone = true) {
        for (const param of this.params) {
            if (setToNone) {
                param.grad = null
            } else if (param.grad !== null) {
                param.grad.data.fill(0)
            }
        }
    }

    step() {
        throw new Error("step() must be implemented by subclasses")
    }
}