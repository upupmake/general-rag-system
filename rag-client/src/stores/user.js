// src/stores/user.js
import {defineStore} from 'pinia'
import commonApi from '@/api/commonApi'

const TOKEN_KEY = 'rag_token'
localStorage.removeItem('rag_user')

export const useUserStore = defineStore('user', {
    state: () => ({
        token: localStorage.getItem(TOKEN_KEY) || '',
        user: null
    }),

    getters: {
        isLogin(state) {
            return !!state.token
        },

        userId(state) {
            return state.user?.id || null
        },

        username(state) {
            return state.user?.username || ''
        },

        role(state) {
            return state.user?.role || null
        },

        roleWeight(state) {
            return state.user?.role?.weight ?? 0
        },
        roleName(state) {
            return state.user?.role?.name || ''
        }
    },

    actions: {
        /**
         * 登录
         * @param {Object} payload { username, password }
         */
        async login(payload) {
            const data = await commonApi.post('/users/login', payload)
            /**
             * 约定返回结构：
             * {
             *   token: 'jwt-token',
             *   user: {
             *     id,
             *     username,
             *     email,
             *     role: { id, name, weight }
             *   }
             * }
             */
            this.token = data.token
            this.user = data.user

            localStorage.setItem(TOKEN_KEY, this.token)
        },

        /**
         * 拉取当前用户信息（用于刷新页面）
         */
        async fetchCurrentUser() {
            this.user = await commonApi.get('/users/me')
        },

        /**
         * 登出
         */
        async logout() {
            try {
                // 调用后端登出接口删除 Redis 中的 token
                await commonApi.post('/users/logout')
            } catch (error) {
                // 即使后端调用失败也继续清除本地信息
                console.error('登出失败:', error)
            } finally {
                // 清除本地存储
                this.token = ''
                this.user = null
                localStorage.removeItem(TOKEN_KEY)
                // 跳转到登录页
                window.location.href = '/login'
            }
        },

        /**
         * 更新用户局部信息（如昵称、邮箱）
         */
        updateUser(partial) {
            this.user = {
                ...this.user,
                ...partial
            }
        }
    }
})
