<script setup>
import {reactive, ref} from 'vue'
import {useRouter} from 'vue-router'
import {message} from 'ant-design-vue'
import md5 from 'crypto-js/md5'
import commonApi from '@/api/commonApi'
import AuthLayout from '@/layouts/AuthLayout.vue'

const router = useRouter()
const loading = ref(false)
const sendCodeLoading = ref(false)
const countdown = ref(0)
let timer = null

const formState = reactive({
  email: '',
  code: '',
  password: '',
  confirmPassword: ''
})

const rules = {
  email: [
    {required: true, message: '请输入邮箱'},
    {type: 'email', message: '请输入正确的邮箱格式'}
  ],
  code: [{required: true, message: '请输入验证码'}],
  password: [{required: true, message: '请输入新密码'}],
  confirmPassword: [
    {required: true, message: '请确认新密码'},
    {
      validator: async (_rule, value) => {
        if (!value || value === formState.password) return Promise.resolve()
        return Promise.reject('两次输入的密码不一致')
      }
    }
  ]
}

const sendCode = async () => {
  if (!formState.email) {
    message.warning('请先输入邮箱')
    return
  }
  sendCodeLoading.value = true
  try {
    await commonApi.post('/users/send-reset-code', {email: formState.email})
    message.success('重置验证码已发送')
    countdown.value = 60
    timer = setInterval(() => {
      countdown.value--
      if (countdown.value <= 0) clearInterval(timer)
    }, 1000)
  } finally {
    sendCodeLoading.value = false
  }
}

const onReset = async () => {
  loading.value = true
  try {
    await commonApi.post('/users/reset-password', {
      email: formState.email,
      code: formState.code,
      newPassword: md5(formState.password).toString()
    })
    message.success('密码重置成功，请重新登录')
    router.push('/login')
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
    <div class="reset-wrapper">
      <div class="auth-header">
        <div class="scene-tag">邮箱安全验证</div>
        <h2 class="title">找回密码</h2>
        <p class="subtitle">通过邮箱验证码重置密码，快速恢复账号访问。</p>
      </div>

      <a-form
        :model="formState"
        :rules="rules"
        layout="vertical"
        @finish="onReset"
        class="auth-form"
      >
        <a-form-item label="邮箱" name="email">
          <a-input v-model:value="formState.email" placeholder="请输入邮箱" size="large"/>
        </a-form-item>

        <a-form-item label="验证码" name="code">
          <div class="code-container">
            <a-input v-model:value="formState.code" placeholder="请输入验证码" size="large"/>
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

        <a-form-item label="新密码" name="password">
          <a-input-password
            v-model:value="formState.password"
            placeholder="请输入新密码"
            size="large"
          />
        </a-form-item>

        <a-form-item label="确认新密码" name="confirmPassword">
          <a-input-password
            v-model:value="formState.confirmPassword"
            placeholder="请再次输入新密码"
            size="large"
          />
        </a-form-item>

        <a-form-item>
          <a-button type="primary" html-type="submit" size="large" block :loading="loading">
            重置密码
          </a-button>
        </a-form-item>

        <p class="hint-text">提交后将立即生效，请使用新密码重新登录。</p>

        <div class="footer-actions">
          <span class="text-gray">想起密码了？</span>
          <a @click="goToLogin">返回登录</a>
        </div>
      </a-form>
    </div>
  </AuthLayout>
</template>

<style scoped>
.reset-wrapper {
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

.hint-text {
  margin: -2px 0 10px;
  text-align: center;
  font-size: 12px;
  line-height: 1.6;
  color: #8a95a8;
}

.footer-actions {
  text-align: center;
  margin-top: 8px;
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
  .reset-wrapper {
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
  .reset-wrapper {
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
