import { MulBackward } from "../autograd/mul-backward.js";
import { elementWiseOp } from "../util.js";

export class MulOp {
    apply(a, b) {
        const output = elementWiseOp(a, b, (x, y) => x * y)

        if (a.requiresGrad || b.requiresGrad) {
            output.requiresGrad = true
            output.gradFn = new MulBackward([a, b])
        }
        return output
    }
}