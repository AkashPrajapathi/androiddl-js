export class CoordinateIterator {

    constructor(shape) {
        this.shape = shape
        this.finished = false
        this.coordinate = new Array(this.shape.length).fill(0)
    }

    hasNext() {
        return !this.finished
    }

    next() {

        if (!this.hasNext()) {
            throw new Error("NoSuchElementException")
        }

        const index = [...this.coordinate]

        this.#advance()

        return index
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