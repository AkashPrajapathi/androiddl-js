import { Tensor } from "../tensor.js";
import { normalizeShape, normalizeSpec, parseSlice } from "../util.js";

export class SliceOp {
    apply(a, ...specs) {
        if (!(a instanceof Tensor)) {
            throw new Error("a must be tensor")
        }

        const rank = a.shape.length

        if (specs.length === 0) {
            specs = Array(rank).fill(":");
        }

        if (specs.length !== rank) {
            throw new Error(
                `Expected ${rank} slice specs, got ${specs.length}`
            );
        }

        if (specs.length === 0) {
            specs = Array(rank).fill(":")
        }

        const shape = []
        const strides = []
        let offset = a.offset

        for (let d = 0; d < rank; d++) {
            const { start, stop, step } = parseSlice(specs[d])
            const normSlice = normalizeSpec(start, stop, step, a.shape[d])

            shape.push(normSlice.length)
            strides.push(a.strides[d] * normSlice.step)

            offset += normSlice.start * a.strides[d]
        }

        const output = Tensor.view(a, shape, strides, offset)

        return output
    }
}