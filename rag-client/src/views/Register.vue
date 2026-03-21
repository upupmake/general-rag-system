<script setup>
import { reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import md5 from 'crypto-js/md5'
import commonApi from '@/api/commonApi'
import AuthLayout from '@/layouts/AuthLayout.vue';

const router = useRouter()
const loading = ref(false)
const sendCodeLoading = ref(false)
const countdown = ref(0)
let timer = null

const formState = reactive({
  username: '',
  password: '',
  email: '',
  code: ''
})

const rules = {
  username: [{ required: true, message: '请输入账号' }],
  password: [{ required: true, message: '请输入密码' }],
  email: [
    { required: true, message: '请输入邮箱' },
    { type: 'email', message: '请输入正确的邮箱格式' }
  ],
  code: [{ required: true, message: '请输入验证码' }]
}

const sendCode = async () => {
  if (!formState.email) {
    message.warning('请先输入邮箱')
    return
  }
  
  sendCodeLoading.value = true
  try {
    await commonApi.post('/users/send-code', { email: formState.email })
    message.success('验证码已发送')
    countdown.value = 60
    timer = setInterval(() => {
      countdown.value--
      if (countdown.value <= 0) {
        clearInterval(timer)
      }
    }, 1000)
  } catch (e) {
    // Error is handled by interceptor usually, or display e.message
    console.error(e)
  } finally {
    sendCodeLoading.value = false
  }
}

const onRegister = async () => {
  loading.value = true
  try {
    await commonApi.post('/users/register', {
      username: formState.username,
      password: md5(formState.password).toString(),
      email: formState.email,
      code: formState.code
    })
    message.success('注册成功，请登录')
    router.push('/login')
  } catch (e) {
    console.error(e)
  } finally {
    loading.value = false
  }
}

const goToLogin = () => {
  router.push('/login')
}
</script>

<template>
  <AuthLayout>
    <div class="register-wrapper">
      <div class="auth-header">
        <div class="scene-tag">快速注册</div>
        <h2 class="title">注册账号</h2>
        <p class="subtitle">创建您的 RAG 账号，立即开始智能检索与对话。</p>
      </div>

      <a-form
        :model="formState"
        :rules="rules"
        layout="vertical"
        @finish="onRegister"
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

        <a-form-item label="邮箱" name="email">
          <a-input
            v-model:value="formState.email"
            placeholder="请输入邮箱"
            size="large"
          />
        </a-form-item>

        <a-form-item label="验证码" name="code">
          <div class="code-container">
            <a-input
              v-model:value="formState.code"
              placeholder="请输入验证码"
              size="large"
            />
            <a-button
              class="code-btn"
              size="large"
              :loading="sendCodeLoading"
              :disabled="countdown > 0"
              @click="sendCode"
            >
              {{ countdown > 0 ? `${countdown}s 后重试` : '获取验证码' }}
            </a-button>
          </div>
        </a-form-item>

        <a-form-item>
          <a-button
            type="primary"
            html-type="submit"
            size="large"
            block
            :loading="loading"
          >
            注册
          </a-button>
        </a-form-item>
        
        <div class="footer-actions">
          <span class="text-gray">已有账号？</span>
          <a @click="goToLogin">立即登录</a>
        </div>
      </a-form>
    </div>
  </AuthLayout>
</template>

<style scoped>
.register-wrapper {
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

.code-container {
  display: flex;
  gap: 8px;
}

.code-btn {
  width: 120px;
  border-radius: 12px;
  border-color: #c9d8f9;
  color: #315ec7;
  font-weight: 600;
}

.code-btn:hover,
.code-btn:focus {
  border-color: #8fb3ff;
  color: #1d4fb8;
  background: #f4f8ff;
}

.footer-actions {
  text-align: center;
  margin-top: 16px;
  font-size: 14px;
}

.footer-actions a {
  color: #1677ff;
  font-weight: 600;
}

.text-gray {
  color: #999;
  margin-right: 8px;
}

@media (max-width: 768px) {
  .register-wrapper {
    max-width: 100%;
    padding: 26px 18px 20px;
    border-radius: 0;
    border: none;
    box-shadow: none;
    background: transparent;
  }

  .title {
    font-size: 26px;
  }

  .subtitle {
    font-size: 13px;
  }

  .code-btn {
    width: 112px;
    flex-shrink: 0;
    padding: 0 10px;
    font-size: 13px;
  }
}

@media (max-width: 420px) {
  .register-wrapper {
    padding: 24px 14px 16px;
  }

  .title {
    font-size: 24px;
  }

  .code-container {
    gap: 6px;
  }

  .code-btn {
    width: 104px;
  }
}
</style>
