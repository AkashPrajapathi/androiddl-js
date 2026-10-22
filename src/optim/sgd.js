import { Optimizer } from "./optimizer.js";

export class SGD extends Optimizer {
    constructor(params, { lr = 0.001, momentum = 0, dampening = 0 } = {}) {
        super(params, { lr, momentum, dampening })
    }

    step() {

        const { lr, momentum, dampening } = this.defaults

        for (const param of this.params) {
            const grad = param.grad

            if (grad === null) continue

            const data = param.data
            const gradData = grad.data

            if (momentum > 0) {

                let state = this.state.get(param)

                if (state === undefined) {
                    state = {
                        buffer: new Float32Array(data.length),
                        initialized: false
                    }

                    this.state.set(param, state)
                }

                const buffer = state.buffer

                for (let i = 0; i < data.length; i++) {
                    if (!state.initialized) {
                        buffer[i] = gradData[i]
                    } else {
                        buffer[i] = momentum * buffer[i] + (1 - dampening) * gradData[i]
                    }

                    data[i] -= lr * buffer[i]
                }

                state.initialized = true

            } else {
                for (let i = 0; i < data.length; i++) {
                    data[i] -= lr * gradData[i]
                }
            }
        }
    }
}