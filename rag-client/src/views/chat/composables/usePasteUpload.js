import {ref} from 'vue'
import {message} from 'ant-design-vue'
import {uploadDocument} from '@/api/kbApi'
import {selectedKb, findKbById} from '@/vars.js'
import {useUserStore} from '@/stores/user'

// 文件类型校验 - 与 KnowledgeDetails.vue 的 acceptExtensions 保持一致
const ACCEPT_EXTENSIONS = '.md,.txt,.pdf,.json,.py,.java,.js,.ts,.vue,.html,.xml,.yml,.sh,.rb,.css,.scss,.jpg,.jpeg,.png,.gif,.bmp,.webp'
const ALLOWED_EXTENSIONS = ACCEPT_EXTENSIONS.split(',')
const MAX_FILE_SIZE_MB = 100
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

// MIME 类型到扩展名的映射（粘贴的剪贴板图片通常无文件名）
const MIME_TO_EXT = {
    'image/png': '.png',
    'image/jpeg': '.jpg',
    'image/gif': '.gif',
    'image/bmp': '.bmp',
    'image/webp': '.webp',
}

export function usePasteUpload() {
    const userStore = useUserStore()
    const uploading = ref(false)

    // 检查当前选中的知识库是否有上传权限（后端 canWriteKb 仅允许 owner 上传）
    const checkKbPermission = () => {
        if (!selectedKb.value) {
            return {ok: false, msg: '请先选择知识库后再粘贴上传'}
        }
        const kb = findKbById(selectedKb.value)
        if (!kb) {
            return {ok: false, msg: '知识库不存在'}
        }
        if (kb.ownerUserId !== userStore.userId) {
            return {ok: false, msg: `您没有向知识库「${kb.name}」上传文件的权限`}
        }
        return {ok: true, kb}
    }

    const getExtension = (file) => {
        if (file.name && file.name.includes('.')) {
            return file.name.substring(file.name.lastIndexOf('.')).toLowerCase()
        }
        return MIME_TO_EXT[file.type] || ''
    }

    // 粘贴的图片通常没有文件名，根据 MIME 类型生成文件名
    const ensureFileName = (file) => {
        if (file.name && file.name.includes('.')) {
            return file
        }
        const ext = MIME_TO_EXT[file.type] || '.png'
        const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
        return new File([file], `paste_${ts}${ext}`, {type: file.type})
    }

    const uploadFile = async (file, kbId) => {
        const formData = new FormData()
        formData.append('files', file, file.name)

        const hide = message.loading(`${file.name} 上传中...`, 0)
        try {
            await uploadDocument(kbId, formData, {})
            hide()
            message.success(`${file.name} 上传成功，正在处理中...`)
        } catch {
            hide()
            // 错误消息已由 commonApi 响应拦截器统一处理
        }
    }

    const handlePaste = async (event) => {
        const items = event.clipboardData?.items
        if (!items) return

        // 只提取普通文件；目录和浏览器无法解析为 File 的条目不触发上传
        const files = []
        for (let i = 0; i < items.length; i++) {
            const item = items[i]
            if (item.kind !== 'file') continue

            const entry = item.webkitGetAsEntry?.()
            if (entry && !entry.isFile) continue

            const file = item.getAsFile()
            if (file) files.push(file)
        }

        // 纯文本、目录或非普通文件粘贴，不拦截
        if (files.length === 0) return

        event.preventDefault()

        // 权限校验
        const perm = checkKbPermission()
        if (!perm.ok) {
            message.warning(perm.msg)
            return
        }

        uploading.value = true
        try {
            for (const file of files) {
                const processedFile = ensureFileName(file)

                // 文件类型校验
                const ext = getExtension(processedFile)
                if (!ALLOWED_EXTENSIONS.includes(ext)) {
                    message.error(`不支持的文件类型: ${processedFile.name}`)
                    continue
                }

                // 文件大小校验
                if (processedFile.size > MAX_FILE_SIZE_BYTES) {
                    const sizeMB = (processedFile.size / (1024 * 1024)).toFixed(2)
                    message.error(`文件 ${processedFile.name} 大小为 ${sizeMB} MB，超过最大限制 ${MAX_FILE_SIZE_MB} MB`)
                    continue
                }

                await uploadFile(processedFile, perm.kb.id)
            }
        } finally {
            uploading.value = false
        }
    }

    return {
        uploading,
        handlePaste,
    }
}
