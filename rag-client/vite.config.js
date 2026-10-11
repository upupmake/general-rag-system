import {defineConfig} from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'
import components from 'unplugin-vue-components/vite';
import {AntDesignVueResolver} from 'unplugin-vue-components/resolvers';
import {AntDesignXVueResolver} from 'ant-design-x-vue/resolver';
import VueJsx from '@vitejs/plugin-vue-jsx'

export default defineConfig({
    plugins: [
        VueJsx(),
        vue(),
        components({
            // ant-design-vue v4 为 CSS-in-JS，无需按需注入样式
            resolvers: [AntDesignVueResolver({importStyle: false}), AntDesignXVueResolver()]
        })
    ],
    resolve: {
        alias: {
            '@': path.resolve('src')
        }
    },
})
