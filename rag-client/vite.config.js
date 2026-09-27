import {defineConfig} from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'
import components from 'unplugin-vue-components/vite';
import {AntDesignXVueResolver} from 'ant-design-x-vue/resolver';
import VueJsx from '@vitejs/plugin-vue-jsx'

export default defineConfig({
    // 相对路径产物: 离线包/子路径部署时无需关心部署前缀
    base: './',
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
