import { Module } from "./module.js"
import { Parameter } from "./parameter.js"

export class Linear extends Module {

    constructor(inFeatures, outFeatures, bias = true, initializer = "xavier") {
        super()

        this.inFeatures = inFeatures
        this.outFeatures = outFeatures

        this.w = new Parameter(this.#initializeParameter(initializer), [outFeatures, inFeatures])
        this.b = bias ? new Parameter(new Float32Array(outFeatures), [outFeatures]) : null
    }

    #initializeParameter(initializer = "xavier") {
        const limit = initializer == "xavier" ? Math.sqrt(6 / (this.inFeatures + this.outFeatures)) : Math.sqrt(6 / this.inFeatures)

        const size = this.inFeatures * this.outFeatures
        const result = new Float32Array(size)

        for (let i = 0; i < size; i++) {
            result[i] = (Math.random() * 2 - 1) * limit
        }
        return result
    }

    forward(input) {
        let out = input.matmul(this.w.transpose(0, 1))
        if (this.b !== null) {
            out = out.add(this.b)
        }
        return out
    }
}