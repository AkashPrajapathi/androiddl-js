import { Module } from "./module.js";

export class Sequential extends Module {
    constructor(...modules) {
        super()

        modules.forEach((module, i) => {
            if (!(module instanceof Module)) {
                throw new TypeError(`Sequential expects Module instances, got index ${i}`)
            }

            this[i] = module
        })
    }

    forward(x) {
        for (const layer of Object.values(this._modules)) {
            x = layer.call(x)
        }

        return x
    }
}