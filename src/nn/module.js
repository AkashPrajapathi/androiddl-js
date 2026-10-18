import { Parameter } from "./parameter.js"

export class Module {

    constructor() {

        this._modules = Object.create(null)
        this._parameters = Object.create(null)

        return new Proxy(this, {

            set(target, prop, value, reciever) {

                if (typeof prop === "string") {

                    delete target._modules[prop]
                    delete target._parameters[prop]

                    if (value instanceof Module) {
                        target._modules[prop] = value
                        delete target._parameters[prop]
                    } else if (value instanceof Parameter) {
                        target._parameters[prop] = value
                        delete target._modules[prop]
                    }
                }

                return Reflect.set(target, prop, value, reciever)

            }, deleteProperty(target, prop) {

                delete target._modules[prop]
                delete target._parameters[prop]

                Reflect.deleteProperty(target, prop)
            }
        })
    }

    parameters() {

        const params = []
        const seenParams = new Set()
        const seenModules = new Set()

        const stack = [this]

        while (stack.length > 0) {

            const module = stack.pop()

            if (seenModules.has(module)) continue

            seenModules.add(module)

            for (const param of Object.values(module._parameters)) {
                if (!seenParams.has(param)) {
                    params.push(param)
                    seenParams.add(param)
                }
            }

            for (const child of Object.values(module._modules)) {
                stack.push(child)
            }
        }

        return params
    }

    call(...inputs) {
        return this.forward(inputs)
    }

    forward(...inputs) {
        throw new Error("farward() must be implemented by subclass")
    }
}