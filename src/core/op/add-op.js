import { elementWiseOp } from "../util.js";

export class AddOp {
    apply(a, b) {
        return elementWiseOp(a, b, (x, y) => x + y)
    }
}