import { defineConfig } from "vite"
import { resolve } from "path"

export default defineConfig({
    build: {
        lib: {
            entry: resolve(__dirname, 'androiddl.js'),
            name: 'AndroidDL',
            fileName: 'androiddl'
        },

        rollupOptions: {
            external: [],
            output: {
                globals: {}
            }
        }
    }
})