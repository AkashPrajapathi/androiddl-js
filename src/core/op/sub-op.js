import { SubBackward } from "../autograd/sub-backward.js";
import { elementWiseOp } from "../util.js";

export class SubOp {
    apply(a, b) {
        const output = elementWiseOp(a, b, (x, y) => x - y)

        if (a.requiresGrad || b.requiresGrad) {
            output.requiresGrad = true
            output.gradFn = new SubBackward([a, b])
        }
        return output
    }
}