
import { SigmoidBackward } from "../autograd/sigmoid-backward.js"
import { Tensor } from "../tensor.js"

export class SigmoidOp {
    apply(a) {
        if (!(a instanceof Tensor)) {
            throw new Error("a must be tensor")
        }

        const result = new Float32Array(a.numel)

        for (let i = 0; i < a.numel; i++) {
            result[i] = 1 / (1 + Math.exp(-a.data[i]))
        }

        const output = new Tensor(result, [...a.shape])

        if (a.requiresGrad) {
            output.requiresGrad = true
            output.gradFn = new SigmoidBackward([a], output)
        }

        return output
    }
}