import axios from 'axios'
import {API_BASE_URL} from '@/consts.js'

export const ADMIN_TOKEN_KEY = 'rag_admin_membership_token'
export const ADMIN_TOKEN_EXPIRES_KEY = 'rag_admin_membership_expires_at'

const http = axios.create({
  baseURL: API_BASE_URL,
  withCredentials: false
})

http.interceptors.request.use(config => {
  const token = localStorage.getItem(ADMIN_TOKEN_KEY)
  if (token) {
    config.headers['X-Admin-Token'] = token
  }
  return config
})

http.interceptors.response.use(
  response => {
    const res = response.data
    if (res?.code !== 200) {
      const error = new Error(res?.message || '操作失败')
      error.status = res?.code
      return Promise.reject(error)
    }
    return res.data
  },
  error => {
    const status = error.response?.status
    const message = error.response?.data?.message || '请求失败，请稍后重试'
    const err = new Error(message)
    err.status = status
    return Promise.reject(err)
  }
)

export function verifyAdminPassword(password) {
  return http.post('/admin/membership/verify', {password})
}

export function listMembers() {
  return http.get('/admin/membership/members')
}

export function grantMembership(account, days) {
  return http.post('/admin/membership/grant', {account, days})
}

export function updateMemberBz(userId, bz, originalBz) {
  return http.put(`/admin/membership/${userId}/bz`, {bz, originalBz})
}

export function logoutAdmin(token) {
  return http.post('/admin/membership/logout', null, {
    headers: token ? {'X-Admin-Token': token} : {}
  })
}
