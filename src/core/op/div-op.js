import { DivBackward } from "../autograd/div-backward.js";
import { elementWiseOp } from "../util.js";

export class DivOp {
    apply(a, b) {
        const output = elementWiseOp(a, b, (x, y) => x / y)

        if (a.requiresGrad || b.requiresGrad) {
            output.requiresGrad = true
            output.gradFn = new DivBackward([a, b])
        }
        return output
    }
}