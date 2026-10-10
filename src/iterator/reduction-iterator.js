
import { computeReducedShape } from "../core/util.js";

export class ReductionIterator {
    constructor(shape, strides, offset, axes, keepdim = false) {
        this.#validate(shape, strides, offset, axes, keepdim);

        this.shape = [...shape];
        this.strides = [...strides];
        this.offset = offset;
        this.axes = [...axes].sort((a, b) => a - b);
        this.keepdim = keepdim;

        this.finished = false;

        this.outputShape = computeReducedShape(
            this.shape,
            this.axes,
            this.keepdim
        );

        this.outputCoordinate = new Array(
            this.outputShape.length
        ).fill(0);

        this.reductionCoordinate = new Array(
            this.axes.length
        ).fill(0);
    }

    #validate(shape, strides, offset, axes, keepdim) {
        // Validate shape.
        if (!Array.isArray(shape)) {
            throw new Error("Shape must be a 1D array");
        }

        if (shape.length === 0) {
            throw new Error("ReductionIterator requires a non-scalar tensor");
        }

        if (!shape.every(
            dim => Number.isInteger(dim) && dim > 0
        )) {
            throw new Error(
                "Shape must contain positive integers"
            );
        }

        // Validate strides.
        if (!Array.isArray(strides)) {
            throw new Error("Strides must be a 1D array");
        }

        if (strides.length !== shape.length) {
            throw new Error(
                "Shape and strides must have the same rank"
            );
        }

        if (!strides.every(
            stride => Number.isInteger(stride) && stride >= 0
        )) {
            throw new Error(
                "Strides must contain non-negative integers"
            );
        }

        // Validate offset.
        if (!Number.isInteger(offset) || offset < 0) {
            throw new Error(
                "Offset must be a non-negative integer"
            );
        }

        // Validate axes.
        if (!Array.isArray(axes)) {
            throw new Error("Axes must be a 1D array");
        }

        if (!axes.every(
            axis =>
                Number.isInteger(axis) &&
                axis >= 0 &&
                axis < shape.length
        )) {
            throw new Error(
                "Axes must contain valid dimension indices"
            );
        }

        if (new Set(axes).size !== axes.length) {
            throw new Error(
                "Axes cannot contain duplicate values"
            );
        }

        // Validate keepdim.
        if (typeof keepdim !== "boolean") {
            throw new Error("keepdim must be a boolean");
        }

        // Validate that the shape and strides can be
        // represented safely by JavaScript numbers.
        const numel = shape.reduce(
            (acc, dim) => acc * dim,
            1
        );

        if (!Number.isSafeInteger(numel)) {
            throw new Error(
                "Tensor contains too many elements"
            );
        }
    }

    hasNext() {
        return !this.finished;
    }

    next() {
        if (!this.hasNext()) {
            throw new Error("No element");
        }

        const inputCoordinate = this.#mergeCoordinate();

        let physicalIndex = this.offset;

        for (let i = 0; i < inputCoordinate.length; i++) {
            physicalIndex +=
                inputCoordinate[i] * this.strides[i];
        }

        const value = {
            outputCoordinate: [...this.outputCoordinate],
            physicalIndex
        };

        this.#advance();

        return value;
    }

    #mergeCoordinate() {
        const input = new Array(this.shape.length).fill(0);

        let outputDim = 0;
        let reductionDim = 0;

        for (let i = 0; i < this.shape.length; i++) {
            if (this.axes.includes(i)) {
                input[i] =
                    this.reductionCoordinate[reductionDim++];
            } else {
                if (this.keepdim) {
                    input[i] = this.outputCoordinate[i];
                } else {
                    input[i] =
                        this.outputCoordinate[outputDim++];
                }
            }
        }

        return input;
    }

    #advanceReductionCoordinate() {
        for (
            let dim = this.reductionCoordinate.length - 1;
            dim >= 0;
            dim--
        ) {
            this.reductionCoordinate[dim]++;

            const axis = this.axes[dim];

            if (
                this.reductionCoordinate[dim] <
                this.shape[axis]
            ) {
                return true;
            }

            this.reductionCoordinate[dim] = 0;
        }

        return false;
    }

    #advance() {
        if (this.#advanceReductionCoordinate()) {
            return;
        }

        for (
            let dim = this.outputCoordinate.length - 1;
            dim >= 0;
            dim--
        ) {
            if (
                this.keepdim &&
                this.axes.includes(dim)
            ) {
                continue;
            }

            this.outputCoordinate[dim]++;

            if (
                this.outputCoordinate[dim] <
                this.outputShape[dim]
            ) {
                return;
            }

            this.outputCoordinate[dim] = 0;
        }

        this.finished = true;
    }
}
