import { elementWiseOp } from "../util.js";

export class SubOp {
    apply(a, b) {
        return elementWiseOp(a, b, (x, y) => x - y)
    }
}