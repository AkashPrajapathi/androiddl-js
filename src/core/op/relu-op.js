import { ReLUBackward } from "../autograd/relu-backward.js"
import { Tensor } from "../tensor.js"

export class ReLUOp {
    apply(a) {
        if (!(a instanceof Tensor)) {
            throw new Error("a must be tensor")
        }

        const result = new Float32Array(a.numel)

        for (let i = 0; i < a.numel; i++) {
            result[i] = Math.max(0, a.data[i])
        }

        const output = new Tensor(result, [...a.shape])

        if (a.requiresGrad) {
            output.requiresGrad = true
            output.gradFn = new ReLUBackward([a])
        }

        return output
    }
}