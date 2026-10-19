import { Module } from "./module.js";

export class Tanh extends Module {
    forward(x) {
        return x.tanh()
    }
}