import { createApp } from 'vue'
import './style.css'
// 移除静态引入，改为动态加载
// import 'github-markdown-css/github-markdown-light.css'
// import 'github-markdown-css/github-markdown-dark.css'
import App from './App.vue'
import { createPinia } from 'pinia'
import 'highlight.js/styles/github.css';
import 'ant-design-vue/dist/reset.css';
import dayjs from 'dayjs';
import 'dayjs/locale/zh-cn';
dayjs.locale('zh-cn');

import router from './router'


const pinia = createPinia()

const app = createApp(App);
app.use(router)
app.use(pinia)
app.mount('#app');
