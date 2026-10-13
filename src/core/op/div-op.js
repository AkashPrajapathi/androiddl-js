import { elementWiseOp } from "../util.js";

export class DivOp {
    apply(a, b) {
        return elementWiseOp(a, b, (x, y) => x / y)
    }
}