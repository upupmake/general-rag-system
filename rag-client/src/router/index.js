// src/router/index.js
import {createRouter, createWebHistory} from 'vue-router'
import MainLayout from '@/layouts/MainLayout.vue'

import {useUserStore} from '@/stores/user'

const routes = [
    {
        path: '/login',
        component: () => import('@/views/Login.vue'),
        meta: {public: true}
    },
    {
        path: '/register',
        component: () => import('@/views/Register.vue'),
        meta: {public: true}
    },
    {
        path: '/forgot-password',
        component: () => import('@/views/ForgotPassword.vue'),
        meta: {public: true}
    },
    {
        path: '/',
        component: MainLayout,
        meta: {requiresAuth: true},
        redirect: '/dashboard',
        children: [
            {
                path: 'dashboard',
                name: 'Dashboard',
                component: () => import('@/views/Dashboard.vue')
            },
            {
                path: 'kb',
                name: 'KnowledgeBases',
                component: () => import('@/views/kb/KnowledgeBases.vue')
            },
            {
                path: 'kb/:kbId',
                name: 'KnowledgeBaseDetail',
                component: () => import('@/views/kb/KnowledgeDetails.vue')
            },
            {
                path: 'chat/new',
                name: 'NewChat',
                component: () => import('@/views/chat/NewChat.vue'),
            },
            {
                path: 'chat/:sessionId',
                name: 'ChatSession',
                component: () => import('@/views/chat/ChatSession.vue'),
            },
            {
                path: 'search',
                name: 'SearchSessions',
                component: () => import('@/views/SearchSessions.vue')
            },
            {
                path: 'access-keys',
                name: 'AccessKeys',
                component: () => import('@/views/AccessKeys.vue')
            },
            {
                path: 'workspaces',
                name: 'WorkspaceManagement',
                component: () => import('@/views/workspace/WorkspaceManagement.vue')
            }
        ]
    }
]

const router = createRouter({
    history: createWebHistory(),
    routes
})

router.beforeEach(async (to, from, next) => {
    const userStore = useUserStore()

    if (to.meta.public) return next()

    if (!userStore.isLogin) {
        return next('/login')
    }

    next()
})

export default router
