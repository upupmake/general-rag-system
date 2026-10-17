import commonApi from './commonApi'
import {API_BASE_URL} from "@/consts.js";
import {events} from '@/events.js'

export function fetchAvailableKbs() {
    return commonApi.get('/kb')
}

export async function createKb(formData) {
    const data = await commonApi.post('/kb', {
        ...formData
    })
    events.emit('knowledgeBasesChanged')
    return data
}

export async function deleteKb(kbId) {
    const data = await commonApi.delete(`/kb/${kbId}`)
    events.emit('knowledgeBasesChanged')
    return data
}

export async function updateKb(kbId, data) {
    const result = await commonApi.put(`/kb/${kbId}`, data)
    events.emit('knowledgeBasesChanged')
    return result
}

// 列出文件
export function listDocuments(kbId) {
    return commonApi.get(`/kb/${kbId}/documents`)
}

// 上传文件
export function uploadDocument(kbId, formData, config = {}) {
    return commonApi.upload(`/kb/${kbId}/documents`, formData, config)
}

// 预览文件 (返回 Blob)
export function previewDocument(kbId, docId) {
    return commonApi.get(`/kb/${kbId}/documents/${docId}/preview`, null, { responseType: 'blob' })
}

// 预览切片
export function listChunks(kbId, docId, page, size) {
    return commonApi.get(`/kb/${kbId}/documents/${docId}/chunks`, { page, size })
}

// 删除文件
export function deleteDocument(kbId, docId) {
    return commonApi.delete(`/kb/${kbId}/documents/${docId}`)
}

// 邀请用户访问知识库
export function inviteUserToKb(kbId, data) {
    return commonApi.post(`/kb/${kbId}/invite`, data)
}

// 获取知识库的被邀请用户列表
export function getInvitedUsers(kbId) {
    return commonApi.get(`/kb/${kbId}/invited-users`)
}

// 移除被邀请用户
export function removeInvitedUser(kbId, userId) {
    return commonApi.delete(`/kb/${kbId}/invited-users/${userId}`)
}