import { ExpBackward } from "../autograd/exp-backward.js"

export class ExpOp {
    apply(a) {
        if (!(a instanceof Tensor)) {
            throw new Error("a must be tensor")
        }

        const result = new Float32Array(a.numel)

        for (let i = 0; i < a.numel; i++) {
            result[i] = Math.exp(a.data[i])
        }

        const output = new Tensor(result, [...a.shape])

        if (a.requiresGrad) {
            output.requiresGrad = true
            output.gradFn = new ExpBackward([a], output)
        }

        return output
    }
}