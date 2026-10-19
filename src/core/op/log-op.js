import { LogBackward } from "../autograd/log-backward.js"


export class LogOp {
    apply(a) {
        if (!(a instanceof Tensor)) {
            throw new Error("a must be tensor")
        }

        const result = new Float32Array(a.numel)

        for (let i = 0; i < a.numel; i++) {
            result[i] = Math.log(a.data[i])
        }

        const output = new Tensor(result, [...a.shape])

        if (a.requiresGrad) {
            output.requiresGrad = true
            output.gradFn = new LogBackward([a])
        }

        return output
    }
}