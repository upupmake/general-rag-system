<script setup>
import {reactive, ref} from 'vue'
import {useRouter} from 'vue-router'
import {message} from 'ant-design-vue'

import {useUserStore} from '@/stores/user'
import md5 from 'crypto-js/md5';
import AuthLayout from '@/layouts/AuthLayout.vue';
import { GithubOutlined } from '@ant-design/icons-vue';

const router = useRouter()
const userStore = useUserStore()

const loading = ref(false)


/**
 * 登录表单
 */
const formState = reactive({
  username: '',
  password: ''
})

/**
 * 表单校验规则
 */
const rules = {
  username: [{required: true, message: '请输入账号'}],
  password: [{required: true, message: '请输入密码'}]
}

/**
 * Mock 登录（开发期）
 * 输入任意账号密码即可
 */
const onLogin = async () => {
  loading.value = true
  try {
    /**
     * 模拟后端返回的数据结构
     * 与 userStore.login 保持一致
     */
    await userStore.login({
      username: formState.username,
      password: md5(formState.password).toString(),
      rememberMe: true
    })

    message.success('登录成功')

    router.push('/dashboard')
  } catch (e) {
    message.error('登录失败')
  } finally {

    loading.value = false
  }
}
</script>

<template>
  <AuthLayout>
    <div class="login-wrapper">
      <div class="auth-header">
        <div class="scene-tag">账号登录</div>
        <h2 class="title">登录账号</h2>
        <p class="subtitle">登录您的 RAG 账号，继续智能检索与对话。</p>
      </div>

      <a-form
          :model="formState"
          :rules="rules"
          layout="vertical"
          @finish="onLogin"
          class="auth-form"
      >
        <a-form-item label="账号" name="username">
          <a-input
              v-model:value="formState.username"
              placeholder="请输入账号"
              size="large"
          />
        </a-form-item>

        <a-form-item label="密码" name="password">
          <a-input-password
              v-model:value="formState.password"
              placeholder="请输入密码"
              size="large"
          />
        </a-form-item>

        <a-form-item>
          <a-button
              type="primary"
              html-type="submit"
              size="large"
              block
              :loading="loading"
          >
            登录
          </a-button>
        </a-form-item>
      </a-form>

      <div class="footer-actions">
        <span class="text-gray">还没有账号？</span>
        <a style="cursor: pointer" @click="router.push('/register')">立即注册</a>
      </div>

      <div class="footer-actions">
        <a style="cursor: pointer" @click="router.push('/forgot-password')">忘记密码？</a>
      </div>
      
      <div class="github-link">
        <a href="https://github.com/cockmake/general-rag-system" target="_blank">
          <github-outlined /> GitHub 开源地址
        </a>
      </div>
    </div>
  </AuthLayout>
</template>

<style scoped>
.login-wrapper {
  width: 100%;
  max-width: 416px;
  padding: 28px 28px 24px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid #e8eefb;
  box-shadow: 0 14px 36px rgba(16, 49, 110, 0.12);
}

.auth-header {
  margin-bottom: 24px;
  text-align: center;
}

.scene-tag {
  display: inline-block;
  margin-bottom: 12px;
  padding: 4px 10px;
  border-radius: 999px;
  background: #eef4ff;
  color: #3f6fd9;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.2px;
}

.title {
  font-size: 30px;
  font-weight: 700;
  margin-bottom: 8px;
  color: #14223b;
  line-height: 1.2;
}

.subtitle {
  color: #627089;
  font-size: 14px;
  line-height: 1.6;
}

.auth-form {
  margin-bottom: 14px;
}

.auth-form :deep(.ant-form-item-label > label) {
  font-size: 13px;
  color: #42526b;
  font-weight: 600;
}

.auth-form :deep(.ant-input-affix-wrapper),
.auth-form :deep(.ant-input) {
  border-radius: 12px;
  border-color: #d7deec;
  transition: all 0.25s ease;
}

.auth-form :deep(.ant-input-affix-wrapper:hover),
.auth-form :deep(.ant-input:hover) {
  border-color: #8fb3ff;
}

.auth-form :deep(.ant-input-affix-wrapper-focused),
.auth-form :deep(.ant-input:focus) {
  border-color: #7ba6ff;
  box-shadow: 0 0 0 3px rgba(22, 119, 255, 0.12);
}

.auth-form :deep(.ant-btn-primary) {
  height: 44px;
  border-radius: 12px;
  font-weight: 600;
  box-shadow: 0 8px 18px rgba(22, 119, 255, 0.28);
}

.footer-actions {
  text-align: center;
  margin-top: 14px;
  font-size: 14px;
}

.text-gray {
  color: #999;
  margin-right: 8px;
}

.footer-actions a {
  color: #1677ff;
  font-weight: 600;
}

.github-link {
  text-align: center;
  margin-top: 20px;
}

.github-link a {
  color: #5f6f88;
  font-size: 13px;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  text-decoration: none;
  transition: all 0.3s;
  background: #f4f7ff;
  padding: 7px 12px;
  border-radius: 999px;
}

.github-link a:hover {
  color: #1677ff;
  background: #edf3ff;
}

@media (max-width: 768px) {
  .login-wrapper {
    max-width: 100%;
    padding: 26px 18px 20px;
    border-radius: 0;
    border: none;
    box-shadow: none;
    background: transparent;
  }

  .auth-header {
    margin-bottom: 22px;
  }

  .title {
    font-size: 26px;
  }

  .subtitle {
    font-size: 13px;
  }
}

@media (max-width: 420px) {
  .login-wrapper {
    padding: 24px 14px 16px;
  }

  .title {
    font-size: 24px;
  }

  .github-link {
    margin-top: 16px;
  }
}
</style>
