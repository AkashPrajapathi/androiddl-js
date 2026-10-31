export class TensorIterator {
    constructor(data, shape, strides, offset) {
        this.data = data;
        this.shape = shape;
        this.strides = strides;
        this.offset = offset;

        this.finished = shape.some(dim => dim === 0);
        this.coordinate = new Int32Array(shape.length);
    }

    hasNext() {
        return !this.finished
    }

    next() {

        if (!this.hasNext()) {
            throw new Error("NoSuchElementException")
        }

        const physicalIndex = this.offset + this.coordinate.reduce((acc, v, i) => acc + v * this.strides[i], 0)

        const val = this.data[physicalIndex]

        this.#advance()

        return val
    }

    getCoordinate() {
        return this.coordinate.slice()
    }

    #advance() {
        for (let dim = this.coordinate.length - 1; dim >= 0; dim--) {

            this.coordinate[dim]++

            if (this.coordinate[dim] < this.shape[dim]) {
                return
            }

            this.coordinate[dim] = 0

        }

        this.finished = true
    }
}