import {defineConfig} from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'
import components from 'unplugin-vue-components/vite';
import {AntDesignXVueResolver} from 'ant-design-x-vue/resolver';
import VueJsx from '@vitejs/plugin-vue-jsx'

export default defineConfig({
    plugins: [
        VueJsx(),
        vue(),
        components({
            resolvers: [AntDesignXVueResolver()]
        })
    ],
    resolve: {
        alias: {
            '@': path.resolve('src')
        }
    },
})
