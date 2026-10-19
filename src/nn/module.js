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

    namedChildren() {
        return Object.entries(this._modules)
    }

    namedParameters() {

        const params = []
        const seenParams = new Set()
        const seenModules = new Set()

        const stack = [{ module: this, prefix: "" }]

        while (stack.length > 0) {

            const { module, prefix } = stack.pop()

            if (seenModules.has(module)) continue

            seenModules.add(module)

            for (const [name, param] of Object.entries(module._parameters)) {

                if (!seenParams.has(param)) {

                    seenParams.add(param)
                    params.push([prefix ? `${prefix}.${name}` : name, param])
                }
            }

            const children = Object.entries(module._modules)

            for (let i = children.length - 1; i >= 0; i--) {
                const [name, child] = children[i]

                stack.push({
                    module: child,
                    prefix: prefix ? `${prefix}.${name}` : name
                })
            }
        }

        return params
    }

    namedModules() {

        const modules = []
        const seen = new Set()
        const stack = [{ module: this, prefix: "" }]

        while (stack.length > 0) {

            const { module, prefix } = stack.pop()

            if (seen.has(module)) continue
            seen.add(module)

            modules.push([prefix, module])

            const children = Object.entries(module._modules)

            for (let i = children.length - 1; i >= 0; i--) {
                const [name, child] = children[i]

                stack.push({
                    module: child,
                    prefix: prefix ? `${prefix}.${name}` : name
                })
            }
        }

        return params
    }

    stateDict() {
        const state = Object.create(null)

        for (const [name, param] of this.namedParameters()) {
            state[name] = {
                shape: [...param.shape],
                data: Array.from(param.data)
            }
        }

        return state
    }


    loadStateDict(state) {
        const params = new Map(this.namedParameters())

        // Pass 1: Validate
        if (state === null || typeof state !== "object" || Array.isArray(state)) {
            throw new Error("State must be an object")
        }

        for (const [name, saved] of Object.entries(state)) {
            const param = params.get(name)

            if (!param) {
                throw new Error(`Unexpected parameter: ${name}`)
            }

            if (
                saved === null ||
                typeof saved !== "object" ||
                !Array.isArray(saved.shape) ||
                !Array.isArray(saved.data)
            ) {
                throw new Error(`Invalid state for parameter: ${name}`)
            }

            if (
                saved.shape.length !== param.shape.length ||
                saved.shape.some((dim, i) => dim !== param.shape[i])
            ) {
                throw new Error(`Shape mismatch for parameter: ${name}`)
            }

            if (saved.data.length !== param.data.length) {
                throw new Error(`Data size mismatch for parameter: ${name}`)
            }

            if (!saved.data.every(Number.isFinite)) {
                throw new Error(`Invalid data values for parameter: ${name}`)
            }
        }

        for (const name of params.keys()) {
            if (!Object.hasOwn(state, name)) {
                throw new Error(`Missing parameter: ${name}`)
            }
        }

        // Pass 2: Copy
        for (const [name, param] of params) {
            param.data.set(state[name].data)
        }

        return this
    }


    call(...inputs) {
        return this.forward(...inputs)
    }

    forward(...inputs) {
        throw new Error("farward() must be implemented by subclass")
    }
}