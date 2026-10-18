<script setup>
import {onMounted, ref, computed, watch} from "vue";
import {onUnmounted} from "vue";
import md from "@/utils/markdown.js";
import DocxPreview from "@/components/DocxPreview.vue";
import dayjs from "dayjs";
import {useRoute, useRouter} from "vue-router";
import {message, theme} from "ant-design-vue";
import {
  LoadingOutlined,
  ArrowLeftOutlined,
  FolderOutlined,
  FileOutlined,
  HomeOutlined,
  MoreOutlined,
  DownOutlined,
  SearchOutlined,
  SettingOutlined,
  TeamOutlined,
  MailOutlined,
  FolderAddOutlined,
  CloudUploadOutlined,
  ReloadOutlined
} from '@ant-design/icons-vue';
import {
  deleteDocument,
  previewDocument,
  listDocuments,
  uploadDocument,
  reprocessDocument,
  listChunks,
  inviteUserToKb,
  getInvitedUsers,
  removeInvitedUser,
  fetchAvailableKbs,
  updateKb
} from "@/api/kbApi.js";
import {useUserStore} from "@/stores/user";

const {useToken} = theme;
const {token} = useToken();
const route = useRoute();
const router = useRouter();
const userStore = useUserStore();
const kbId = route.params.kbId;
const fileList = ref([]);
const uploading = ref(false);
const currentKb = ref(null);
const currentPath = ref([]);
const searchKeyword = ref('');
const processingCount = computed(() => fileList.value.filter(file => file.status === 'processing').length);
const failedCount = computed(() => fileList.value.filter(file => file.status === 'failed').length);
const processedNotice = ref('');
let documentPollTimer = null;
let noticeTimer = null;
let documentsFetching = false;
let pendingDocumentFetch = false;
let hadProcessing = false;

const currentPathString = computed(() => {
  return currentPath.value.join('/');
});

const getRelativePath = (fullPath) => {
  const prefix = currentPathString.value ? currentPathString.value + '/' : '';
  if (!fullPath.startsWith(prefix)) return null;
  return fullPath.substring(prefix.length);
};

const displayList = computed(() => {
  const list = [];
  const folders = new Set();

  if (!fileList.value) return [];

  const keyword = searchKeyword.value.trim().toLowerCase();
  const pathMatches = (fullName) => !keyword || fullName.toLowerCase().includes(keyword);

  // Sort files first to ensure consistent order
  const sortedFiles = [...fileList.value].sort((a, b) => (a.fileName || '').localeCompare(b.fileName || ''));

  sortedFiles.forEach(file => {
    // Normalize path separators just in case
    const fileName = (file.fileName || '').replace(/\\/g, '/');
    const rel = getRelativePath(fileName);
    if (rel === null) return; // Not in current folder

    // Ignore if it's exactly the folder itself (shouldn't happen with files but just in case)
    if (rel === '') return;

    const parts = rel.split('/');
    if (parts.length > 1) {
      // It's in a subfolder
      const folderName = parts[0];
      if (!folders.has(folderName)) {
        folders.add(folderName);

        // Calculate folder stats
        const prefix = (currentPathString.value ? currentPathString.value + '/' : '') + folderName + '/';
        const filesInFolder = fileList.value.filter(f => (f.fileName || '').replace(/\\/g, '/').startsWith(prefix));

        // 搜索时仅保留包含匹配文件的目录，保留原始目录结构
        if (keyword && !filesInFolder.some(f => pathMatches((f.fileName || '').replace(/\\/g, '/')))) return;

        let totalSize = 0;
        let status = 'ready';
        let hasProcessing = false;
        let hasFailed = false;
        let latestCreatedAt = null;

        filesInFolder.forEach(f => {
          totalSize += (f.fileSize || 0);
          if (f.status === 'failed') hasFailed = true;
          if (f.status === 'processing') hasProcessing = true;
          if (f.createdAt && (!latestCreatedAt || new Date(f.createdAt).getTime() > new Date(latestCreatedAt).getTime())) {
            latestCreatedAt = f.createdAt;
          }
        });

        if (hasFailed) status = 'failed';
        else if (hasProcessing) status = 'processing';

        list.push({
          id: 'folder-' + folderName, // unique key for UI
          fileName: folderName,
          isFolder: true,
          fileSize: totalSize,
          status: status,
          createdAt: latestCreatedAt,
        });
      }
    } else {
      // It's a file in current folder
      if (!pathMatches(fileName)) return; // 搜索时过滤不匹配的文件
      list.push({
        ...file,
        isFolder: false
      });
    }
  });

  // Sort folders first, then files
  return list.sort((a, b) => {
    if (a.isFolder && !b.isFolder) return -1;
    if (!a.isFolder && b.isFolder) return 1;
    return a.fileName.localeCompare(b.fileName);
  });
});

const isSearching = computed(() => searchKeyword.value.trim() !== '');

const enterFolder = (folderName) => {
  currentPath.value.push(folderName);
};

const navToLevel = (index) => {
  if (index === -1) {
    currentPath.value = [];
  } else {
    currentPath.value = currentPath.value.slice(0, index + 1);
  }
};

const handleDeleteFolder = async (folderName) => {
  const prefix = (currentPathString.value ? currentPathString.value + '/' : '') + folderName + '/';
  // Find all files that start with this prefix
  const filesToDelete = fileList.value.filter(f => {
    const fn = (f.fileName || '').replace(/\\/g, '/');
    return fn.startsWith(prefix);
  });

  if (filesToDelete.length === 0) {
    message.warning('空文件夹或无法找到文件');
    return;
  }

  const hide = message.loading(`正在删除 ${filesToDelete.length} 个文件...`, 0);
  try {
    // Execute sequentially to avoid overwhelming server or hitting rate limits
    // Or parallel with limit. 
    // Since we don't have a batch delete API, we loop.
    for (const file of filesToDelete) {
      await deleteDocument(kbId, file.id);
    }
    message.success('文件夹删除成功');
    fetchDocuments();
  } catch (e) {
    console.error(e);
    message.error('部分文件删除失败，请重试');
    fetchDocuments(); // Refresh to see what's left
  } finally {
    hide();
  }
};

const goBack = () => {
  router.push('/kb');
};

// 判断是否是拥有者
const isOwner = computed(() => {
  return currentKb.value && currentKb.value.ownerUserId === userStore.userId;
});

// Settings related refs
const settingsModalVisible = ref(false);
const settingsForm = ref({
  name: '',
  description: '',
  systemPrompt: '',
  visibility: 'private'
});
const settingsSubmitting = ref(false);

const openSettingsModal = () => {
  if (currentKb.value) {
    settingsForm.value = {
      name: currentKb.value.name,
      description: currentKb.value.description,
      systemPrompt: currentKb.value.systemPrompt || '',
      visibility: currentKb.value.visibility || 'private'
    };
    settingsModalVisible.value = true;
  }
};

const handleSettingsSubmit = async () => {
  settingsSubmitting.value = true;
  try {
    await updateKb(kbId, settingsForm.value);
    message.success('更新成功');
    settingsModalVisible.value = false;
    fetchKbInfo(); // Refresh info
  } catch (e) {
    console.error('Update failed', e);
  } finally {
    settingsSubmitting.value = false;
  }
};

// 判断是否是私有知识库且是拥有者
const canInvite = computed(() => {
  return currentKb.value && currentKb.value.visibility === 'private' && isOwner.value;
});

// 页头展示的可见性信息
const visibilityLabel = computed(() => {
  const v = currentKb.value?.visibility;
  if (v === 'public') return '公开';
  if (v === 'shared') return '工作空间共享';
  return '私有';
});

const visibilityTagColor = computed(() => {
  const v = currentKb.value?.visibility;
  if (v === 'public') return 'orange';
  if (v === 'shared') return 'green';
  return 'default';
});

const formatFileDate = (value) => {
  if (!value) return '—';
  return dayjs(value).format('YYYY-MM-DD');
};

// Preview related refs
const previewVisible = ref(false);
const previewContent = ref('');
const previewType = ref('');
const previewTitle = ref('预览');
const docxBlob = ref(null);

// Invite related refs
const inviteModalVisible = ref(false);
const inviteFormData = ref({
  kbId: kbId,
  userIdentifier: ''
});
const inviteSubmitting = ref(false);

// Invited users related refs
const invitedUsersModalVisible = ref(false);
const invitedUsers = ref([]);
const loadingInvitedUsers = ref(false);

// Upload progress related refs
const uploadProgressModalVisible = ref(false);
const uploadProgressList = ref([]);
const activeUploadsCount = ref(0);

// Download/Preview progress related refs
const downloadProgress = ref({
  visible: false,
  title: ''
});

watch(uploadProgressModalVisible, (val) => {
  if (!val && activeUploadsCount.value === 0) {
    uploadProgressList.value = [];
  }
});

const systemPromptPlaceholder = `你是一个专业的AI助手。基于提供的文档和对话历史回答用户问题。

要求：
1. 优先基于文档内容作答，文档是主要信息来源
2. 如果文档不足以完整回答，结合对话历史进行推理或明确说明
3. 文档中的信息为切片信息，可能语义并不连贯或存在错误，你需要抽取或推理相关信息`;

const acceptExtensions = ".md,.txt,.pdf,.docx,.json,.py,.java,.js,.ts,.vue,.html,.xml,.yml,.sh,.rb,.css,.scss,.jpg,.jpeg,.png,.gif,.bmp,.webp";

const invitedUsersColumns = [
  {title: '用户名', dataIndex: 'username', key: 'username'},
  {title: '邮箱', dataIndex: 'email', key: 'email'},
  {title: '邀请人', dataIndex: 'grantedByUsername', key: 'grantedByUsername'},
  {title: '邀请时间', dataIndex: 'grantedAt', key: 'grantedAt'},
  {title: '操作', key: 'action', width: 100}
];

const columns = [
  {title: '', key: 'icon', width: 50, align: 'center'},
  {title: '文件名', dataIndex: 'fileName', key: 'fileName'},
  {title: '大小', dataIndex: 'fileSize', key: 'fileSize'},
  {title: '状态', dataIndex: 'status', key: 'status'}, // processing, ready, failed
  {
    title: '上传时间',
    dataIndex: 'createdAt',
    key: 'createdAt',
    sorter: (a, b, sortOrder) => {
      if (a.isFolder && !b.isFolder) return sortOrder === 'descend' ? 1 : -1;
      if (!a.isFolder && b.isFolder) return sortOrder === 'descend' ? -1 : 1;
      return new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();
    },
    sortDirections: ['descend', 'ascend']
  },
  {title: '操作', key: 'action', width: 220},
];

const stopDocumentPolling = () => {
  if (documentPollTimer) {
    clearTimeout(documentPollTimer);
    documentPollTimer = null;
  }
};

const scheduleDocumentPolling = () => {
  stopDocumentPolling();
  if (processingCount.value === 0) {
    if (hadProcessing) {
      hadProcessing = false;
      processedNotice.value = failedCount.value > 0 ? 'warning' : 'success';
      clearTimeout(noticeTimer);
      noticeTimer = setTimeout(() => {
        processedNotice.value = '';
      }, 6000);
    }
    return;
  }
  hadProcessing = true;
  processedNotice.value = '';
  documentPollTimer = setTimeout(fetchDocuments, 3000);
};

const syncUploadProgress = () => {
  uploadProgressList.value.forEach(item => {
    if (item.status !== 'done' || !item.processing) return;
    const file = fileList.value.find(doc =>
        (doc.fileName || '').split('/').pop() === item.name);
    if (file?.status === 'ready') {
      item.processing = false;
    } else if (file?.status === 'failed') {
      item.status = 'error';
      item.processing = false;
      item.error = file.errorMessage || '文档处理失败';
    }
  });
};

const fetchDocuments = async () => {
  if (documentsFetching) {
    pendingDocumentFetch = true;
    return;
  }
  documentsFetching = true;
  try {
    fileList.value = await listDocuments(kbId);
    syncUploadProgress();
  } catch (e) {
    console.error('Failed to load documents', e);
  } finally {
    documentsFetching = false;
    scheduleDocumentPolling();
    if (pendingDocumentFetch) {
      pendingDocumentFetch = false;
      fetchDocuments();
    }
  }
};

// 手动刷新列表：只重新拉取文档，保留当前目录与路径过滤条件
const refreshing = ref(false);
const handleRefresh = async () => {
  if (refreshing.value) return;
  refreshing.value = true;
  try {
    await fetchDocuments();
  } finally {
    refreshing.value = false;
  }
};

const handleReprocess = async (record) => {
  try {
    await reprocessDocument(kbId, record.id);
    message.success('已重新提交处理');
    fetchDocuments();
  } catch (e) {
    console.error('Reprocess failed', e);
  }
};

// 获取当前知识库信息
const fetchKbInfo = async () => {
  try {
    const data = await fetchAvailableKbs();
    // 在所有分类中查找当前知识库
    const allKbs = [
      ...(data.private || []),
      ...(data.shared || []),
      ...(data.public || []),
      ...(data.invited || [])
    ];
    currentKb.value = allKbs.find(kb => kb.id == kbId);
  } catch (e) {
    console.error('Failed to fetch KB info', e);
  }
};

// 1. 上传逻辑
// 用于追踪文件夹上传的文件数量
const folderUploadFileSet = ref(new Set());
const folderUploadChecked = ref(false);
const folderUploadBlocked = ref(false); // 标记本次文件夹上传是否已被阻止
const MAX_FOLDER_FILES = 300;
const MAX_FILE_SIZE_MB = 100;
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;
const sizeUploadBlocked = ref(false); // 标记本次上传是否因文件过大而被阻止

// 自动排除的文件夹列表
const EXCLUDED_FOLDERS = [
  'node_modules',
  '.git',
  '.venv',
  'venv',
  'dist',
  'build',
  '.idea',
  '.vscode',
  '__pycache__',
  '.pytest_cache',
  '.mypy_cache',
  'target',
  'vendor',
  '.gradle',
  '.next',
  '.nuxt'
];

// 检查文件路径是否在排除列表中
const isFileExcluded = (filePath) => {
  const pathParts = filePath.split('/');
  return EXCLUDED_FOLDERS.some(excluded => pathParts.includes(excluded));
};

const beforeUpload = (file, fileList) => {
  // ① 文件类型校验（逐文件）
  const extension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
  const allowed = acceptExtensions.split(',');
  if (!allowed.includes(extension)) {
    message.error(`不支持的文件类型: ${file.name}`);
    return false;
  }

  // ② 文件夹上传逻辑
  if (file.webkitRelativePath) {
    // 自动排除开发环境目录
    if (isFileExcluded(file.webkitRelativePath)) {
      return false;
    }

    if (folderUploadBlocked.value) {
      return false;
    }

    // 首次检测到文件夹上传时，统计有效文件数量
    if (!folderUploadChecked.value) {
      folderUploadFileSet.value.clear();

      if (fileList && fileList.length > 0) {
        let totalFiles = 0;
        let excludedFiles = 0;
        let oversizedFile = null;

        fileList.forEach(f => {
          if (f.webkitRelativePath) {
            if (isFileExcluded(f.webkitRelativePath)) {
              excludedFiles++;
            } else {
              folderUploadFileSet.value.add(f.uid);
              totalFiles++;
              if (!oversizedFile && f.size > MAX_FILE_SIZE_BYTES) {
                oversizedFile = f;
              }
            }
          }
        });

        console.log(`文件夹递归扫描完成，有效文件数: ${totalFiles}，已排除文件数: ${excludedFiles}`);

        if (excludedFiles > 0) {
          message.info(`已自动排除 ${excludedFiles} 个开发环境文件（如 node_modules、.git 等）`, 3);
        }

        if (oversizedFile) {
          const sizeMB = (oversizedFile.size / (1024 * 1024)).toFixed(2);
          message.error(`文件 ${oversizedFile.name} 大小为 ${sizeMB} MB，超过最大限制 ${MAX_FILE_SIZE_MB} MB，整批次上传已取消`, 5);
          folderUploadChecked.value = false;
          folderUploadBlocked.value = true;
          folderUploadFileSet.value.clear();

          setTimeout(() => {
            folderUploadBlocked.value = false;
          }, 1000);

          return false;
        }

        if (totalFiles > MAX_FOLDER_FILES) {
          message.warning(`检测到有效文件数量为 ${totalFiles}，超过最大限制 ${MAX_FOLDER_FILES}。请检查是否包含不必要的大文件或文件夹，最大文件数量为 ${MAX_FOLDER_FILES}。`, 5);
          folderUploadChecked.value = false;
          folderUploadBlocked.value = true;
          folderUploadFileSet.value.clear();

          setTimeout(() => {
            folderUploadBlocked.value = false;
          }, 1000);

          return false;
        }
      }

      folderUploadChecked.value = true;
    }

    if (folderUploadFileSet.value.size > 0) {
      folderUploadFileSet.value.delete(file.uid);
      if (folderUploadFileSet.value.size === 0) {
        folderUploadChecked.value = false;
      }
    }
  } else {
    // 普通文件上传，重置文件夹上传状态
    folderUploadChecked.value = false;
    folderUploadFileSet.value.clear();
    folderUploadBlocked.value = false;

    // 文件大小校验（整批次拒绝）
    if (sizeUploadBlocked.value) {
      return false;
    }
    if (file.size > MAX_FILE_SIZE_BYTES) {
      const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
      message.error(`文件 ${file.name} 大小为 ${sizeMB} MB，超过最大限制 ${MAX_FILE_SIZE_MB} MB，整批次上传已取消`, 5);
      sizeUploadBlocked.value = true;
      setTimeout(() => {
        sizeUploadBlocked.value = false;
      }, 1000);
      return false;
    }
  }

  return true;
};

const MAX_CONCURRENT_UPLOADS = 5;
const concurrentUploads = ref(0);
const uploadQueue = [];

const processQueue = async () => {
  if (uploadQueue.length === 0 || concurrentUploads.value >= MAX_CONCURRENT_UPLOADS) {
    return;
  }

  while (uploadQueue.length > 0 && concurrentUploads.value < MAX_CONCURRENT_UPLOADS) {
    const task = uploadQueue.shift();
    concurrentUploads.value++;

    const {formData, file, onSuccess, onError} = task;

    try {
      await uploadDocument(kbId, formData, {
        onUploadProgress: (progressEvent) => {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          const item = uploadProgressList.value.find(item => item.uid === file.uid);
          if (item) {
            item.percent = percent;
          }
        }
      });

      // Success
      const item = uploadProgressList.value.find(item => item.uid === file.uid);
      if (item) {
        item.status = 'done';
        item.percent = 100;
        item.processing = true;
      }
      message.success(`${file.name} 已上传，正在处理`);
      onSuccess(null, file);
      fetchDocuments();
    } catch (err) {
      // Error
      const item = uploadProgressList.value.find(item => item.uid === file.uid);
      if (item) {
        item.status = 'error';
        item.error = err.message || '上传失败';
      }
      onError(err);
    } finally {
      concurrentUploads.value--;
      activeUploadsCount.value--;
      if (activeUploadsCount.value === 0) {
        uploading.value = false;
      }
      processQueue();
    }
  }
};

const customRequest = async (options) => {
  const {file, onSuccess, onError} = options;

  // Initialize file in progress list
  const fileItem = {
    uid: file.uid,
    name: file.name,
    status: 'uploading',
    percent: 0,
    processing: false,
    error: null
  };

  // Add to list
  uploadProgressList.value.push(fileItem);
  uploadProgressModalVisible.value = true;
  activeUploadsCount.value++;
  uploading.value = true;

  const formData = new FormData();
  // Use webkitRelativePath if available (folder upload), otherwise fallback to name.
  // We explicitly set the filename in formData to include the path.
  const prefix = currentPathString.value ? currentPathString.value + '/' : '';
  const relativePath = file.webkitRelativePath || file.name;
  const fullPath = prefix + relativePath;
  formData.append('files', file, fullPath);

  // Add to queue and process
  uploadQueue.push({formData, file, onSuccess, onError});
  processQueue();
};

// 2. 预览逻辑
const clearPreview = () => {
  if (['pdf', 'image'].includes(previewType.value) && previewContent.value) {
    URL.revokeObjectURL(previewContent.value);
  }
  previewContent.value = '';
  previewType.value = '';
  docxBlob.value = null;
};

const handlePreview = async (record) => {
  clearPreview();
  downloadProgress.value = {visible: true, title: '正在加载预览...'};
  try {
    const blob = await previewDocument(kbId, record.id);
    downloadProgress.value.visible = false;

    const fileName = record.fileName ? record.fileName.toLowerCase() : '';
    previewTitle.value = record.fileName || '文件预览';

    if (fileName.endsWith('.pdf')) {
      previewType.value = 'pdf';
      previewContent.value = URL.createObjectURL(blob);
    } else if (fileName.match(/\.(jpeg|jpg|png|gif|bmp|webp)$/)) {
      previewType.value = 'image';
      previewContent.value = URL.createObjectURL(blob);
    } else if (fileName.endsWith('.docx')) {
      previewType.value = 'docx';
      docxBlob.value = blob;
    } else if (fileName.endsWith('.md')) {
      previewType.value = 'markdown';
      previewContent.value = await blob.text();
    } else {
      previewType.value = 'text';
      previewContent.value = await blob.text();
    }
    previewVisible.value = true;
  } catch (e) {
    clearPreview();
    console.error('Preview failed', e);
    message.error('预览失败');
    downloadProgress.value.visible = false;
  }
};

const handlePreviewCancel = () => {
  previewVisible.value = false;
  clearPreview();
};

// 3. 删除逻辑
const handleDelete = async (record) => {
  await deleteDocument(kbId, record.id);
  message.success('已删除');
  fetchDocuments();
};

// 4. 下载逻辑
const handleDownload = async (record) => {
  downloadProgress.value = {visible: true, title: '正在准备下载...'};
  try {
    const blob = await previewDocument(kbId, record.id);
    downloadProgress.value.visible = false;

    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = record.fileName || 'download';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
    message.success('开始下载');
  } catch (e) {
    console.error('Download failed', e);
    message.error('下载失败');
    downloadProgress.value.visible = false;
  }
};

// 6. 切片预览逻辑
const chunksDrawerVisible = ref(false);
const chunksList = ref([]);
const chunksLoading = ref(false);
const currentDocId = ref(null);
const currentPage = ref(1);
const pageSize = ref(10);
const hasMoreChunks = ref(true);

const handlePreviewChunks = (record) => {
  currentDocId.value = record.id;
  currentPage.value = 1;
  chunksList.value = [];
  hasMoreChunks.value = true;
  chunksDrawerVisible.value = true;
  fetchChunks();
};

const fetchChunks = async () => {
  if (!currentDocId.value) return;
  chunksLoading.value = true;
  try {
    const res = await listChunks(kbId, currentDocId.value, currentPage.value, pageSize.value);
    if (res && res.records) {
      chunksList.value.push(...res.records);
      // Determine if there are more chunks
      if (res.records.length < pageSize.value || chunksList.value.length >= res.total) {
        hasMoreChunks.value = false;
      }
    } else {
      hasMoreChunks.value = false;
    }
  } catch (e) {
    console.error("Failed to load chunks", e);
  } finally {
    chunksLoading.value = false;
  }
};

const loadMoreChunks = () => {
  currentPage.value++;
  fetchChunks();
};

const closeChunksDrawer = () => {
  chunksDrawerVisible.value = false;
  chunksList.value = [];
};

// 7. 邀请用户逻辑
const showInviteModal = () => {
  inviteFormData.value.userIdentifier = '';
  inviteModalVisible.value = true;
};

const handleInviteSubmit = async () => {
  if (!inviteFormData.value.userIdentifier || !inviteFormData.value.userIdentifier.trim()) {
    message.warning('请输入用户名或邮箱');
    return;
  }

  inviteSubmitting.value = true;
  try {
    await inviteUserToKb(kbId, inviteFormData.value);
    message.success('邀请成功');
    inviteModalVisible.value = false;
  } catch (e) {
    console.error('Invite failed', e);
  } finally {
    inviteSubmitting.value = false;
  }
};

// 8. 查看被邀请用户列表
const showInvitedUsersModal = async () => {
  invitedUsersModalVisible.value = true;
  loadingInvitedUsers.value = true;

  try {
    invitedUsers.value = await getInvitedUsers(kbId);
  } catch (e) {
    console.error('Failed to load invited users', e);
  } finally {
    loadingInvitedUsers.value = false;
  }
};

// 9. 移除被邀请用户
const handleRemoveInvitedUser = async (record) => {
  try {
    await removeInvitedUser(kbId, record.userId);
    message.success('已移除');
    // 刷新列表
    showInvitedUsersModal();
  } catch (e) {
    console.error('Remove failed', e);
  }
};

const isMobile = ref(false)

const checkIsMobile = () => {
  isMobile.value = window.innerWidth <= 768
}

onMounted(() => {
  checkIsMobile()
  window.addEventListener('resize', checkIsMobile)
  fetchKbInfo();
  fetchDocuments();
});

onUnmounted(() => {
  window.removeEventListener('resize', checkIsMobile)
  stopDocumentPolling()
  clearTimeout(noticeTimer)
  clearPreview()
})
</script>

<template>
  <div class="kb-page">
    <div class="kb-header">
      <div class="kb-title-container">
        <a-button class="kb-back-btn" @click="goBack" type="text" shape="circle">
          <template #icon>
            <arrow-left-outlined/>
          </template>
        </a-button>
        <div class="kb-title-wrap">
          <h2 class="kb-title">{{ currentKb ? currentKb.name : '' }}</h2>
          <div class="kb-meta">
            <span>文档管理</span>
            <span class="kb-meta-dot">·</span>
            <span>共 {{ fileList.length }} 个文件</span>
            <a-tag v-if="currentKb" class="kb-visibility-tag" :color="visibilityTagColor">{{ visibilityLabel }}</a-tag>
          </div>
        </div>
      </div>
      <div class="kb-actions">
        <a-button v-if="isOwner" @click="openSettingsModal">
          <template #icon><setting-outlined/></template>
          <span v-if="!isMobile">设置</span>
        </a-button>
        <a-button v-if="canInvite" @click="showInvitedUsersModal">
          <template #icon><team-outlined/></template>
          <span v-if="!isMobile">被邀请用户</span>
        </a-button>
        <a-button v-if="canInvite" @click="showInviteModal">
          <template #icon><mail-outlined/></template>
          <span v-if="!isMobile">邀请用户</span>
        </a-button>
        <a-upload
            :customRequest="customRequest"
            :showUploadList="false"
            :accept="acceptExtensions"
            :before-upload="beforeUpload"
            directory
            multiple>
          <a-button>
            <template #icon><folder-add-outlined/></template>
            <span v-if="!isMobile">上传文件夹</span>
          </a-button>
        </a-upload>
        <a-upload
            :customRequest="customRequest"
            :showUploadList="false"
            :accept="acceptExtensions"
            :before-upload="beforeUpload"
            multiple>
          <a-button type="primary" :loading="uploading">
            <template #icon><cloud-upload-outlined/></template>
            <span v-if="!isMobile">上传文档</span>
          </a-button>
        </a-upload>
      </div>
    </div>

    <a-alert
        v-if="processingCount > 0"
        type="info"
        show-icon
        class="kb-processing-alert"
        :message="`${processingCount} 个文件正在处理，完成后即可提问`"
        description="处理期间可以离开本页，状态会自动更新。"
    />
    <a-alert
        v-else-if="processedNotice"
        :type="processedNotice"
        show-icon
        class="kb-processing-alert"
        :message="processedNotice === 'success' ? '文档处理完成，现在可以提问' : '部分文档处理失败，请查看失败原因并重试'"
    />

    <!-- 搜索与面包屑 -->
    <div class="kb-toolbar">
      <div class="kb-breadcrumb">
        <a-breadcrumb separator=">">
          <a-breadcrumb-item>
            <a @click="navToLevel(-1)">
              <home-outlined/>
              根目录</a>
          </a-breadcrumb-item>
          <a-breadcrumb-item v-for="(folder, index) in currentPath" :key="index">
            <a @click="navToLevel(index)">{{ folder }}</a>
          </a-breadcrumb-item>
        </a-breadcrumb>
      </div>
      <div class="kb-toolbar-right">
        <a-input
            v-model:value="searchKeyword"
            allow-clear
            placeholder="按路径过滤文件"
            class="kb-search-input">
          <template #prefix><search-outlined/></template>
        </a-input>
        <a-tooltip title="刷新列表">
          <a-button class="kb-refresh-btn" :loading="refreshing" @click="handleRefresh">
            <template #icon><reload-outlined/></template>
            <span v-if="!isMobile">刷新</span>
          </a-button>
        </a-tooltip>
      </div>
    </div>

    <!-- 移动端文件列表 -->
    <div v-if="isMobile" class="kb-mobile-list">
      <a-empty
          v-if="displayList.length === 0"
          class="kb-mobile-empty"
          :description="isSearching ? '未找到匹配的文件' : '当前目录暂无文件'"/>
      <div
          v-for="record in displayList"
          :key="record.id"
          class="kb-mobile-card">
        <button
            type="button"
            class="kb-mobile-main"
            :aria-label="record.isFolder ? `打开文件夹 ${record.fileName}` : `预览 ${record.fileName.split('/').pop()}`"
            @click="record.isFolder ? enterFolder(record.fileName) : handlePreview(record)">
          <span class="kb-mobile-icon" :class="{'is-folder': record.isFolder}">
            <folder-outlined v-if="record.isFolder"/>
            <file-outlined v-else/>
          </span>
          <span class="kb-mobile-body">
            <span class="kb-mobile-name">{{ record.isFolder ? record.fileName : record.fileName.split('/').pop() }}</span>
            <span class="kb-mobile-meta">
              <span>{{ (record.fileSize / (1024 * 1024)).toFixed(2) }} MB</span>
              <span class="kb-meta-dot">·</span>
              <span>{{ formatFileDate(record.createdAt) }}</span>
              <a-tag v-if="record.status === 'processing'" color="blue" class="kb-mobile-status">
                <loading-outlined/>
                向量化中
              </a-tag>
              <a-tag v-else-if="record.status === 'failed'" color="red" class="kb-mobile-status">失败</a-tag>
            </span>
            <span
                v-if="!record.isFolder && record.status === 'failed' && record.errorMessage"
                class="kb-mobile-error">
              {{ record.errorMessage }}
            </span>
          </span>
        </button>
        <a-dropdown :trigger="['click']" placement="bottomRight">
          <a-button type="text" size="small" class="kb-mobile-more" @click.stop>
            <more-outlined/>
          </a-button>
          <template #overlay>
            <a-menu>
              <template v-if="record.isFolder">
                <a-menu-item @click="enterFolder(record.fileName)">打开文件夹</a-menu-item>
                <a-menu-item danger>
                  <a-popconfirm
                      title="确定要删除这个文件夹及其所有内容吗？"
                      ok-text="确定"
                      cancel-text="取消"
                      @confirm="handleDeleteFolder(record.fileName)">
                    <span>删除文件夹</span>
                  </a-popconfirm>
                </a-menu-item>
              </template>
              <template v-else>
                <a-menu-item @click="handlePreview(record)">预览</a-menu-item>
                <a-menu-item @click="handleDownload(record)">下载</a-menu-item>
                <a-menu-item @click="handlePreviewChunks(record)">预览切片</a-menu-item>
                <a-menu-item v-if="record.status === 'failed'" @click="handleReprocess(record)">重试处理</a-menu-item>
                <a-menu-divider/>
                <a-menu-item danger>
                  <a-popconfirm
                      title="确定要删除这个文件吗？"
                      ok-text="确定"
                      cancel-text="取消"
                      @confirm="handleDelete(record)">
                    <span>删除文件</span>
                  </a-popconfirm>
                </a-menu-item>
              </template>
            </a-menu>
          </template>
        </a-dropdown>
      </div>
    </div>

    <a-table v-else class="kb-table" :columns="columns" :data-source="displayList" row-key="id" :pagination="false"
             :scroll="{ x: 800, y: 'calc(100vh - 250px)' }"
             :locale="{ emptyText: isSearching ? '未找到匹配的文件' : '当前目录暂无文件' }">
      <template #bodyCell="{ column, record }">
        <template v-if="column.key === 'icon'">
          <folder-outlined v-if="record.isFolder" class="kb-cell-icon kb-cell-icon-folder"/>
          <file-outlined v-else class="kb-cell-icon"/>
        </template>
        <template v-if="column.key === 'fileName'">
          <a-tooltip v-if="record.isFolder" :title="record.fileName">
            <a class="kb-file-link kb-file-link-folder" @click="enterFolder(record.fileName)">
              {{ record.fileName }}
            </a>
          </a-tooltip>
          <a-tooltip v-else :title="record.fileName.split('/').pop()">
            <span class="kb-file-name">{{ record.fileName.split('/').pop() }}</span>
          </a-tooltip>
        </template>
        <template v-if="column.key === 'fileSize'">
          <span>{{ (record.fileSize / (1024 * 1024)).toFixed(2) }} MB</span>
        </template>
        <template v-if="column.key === 'status'">
          <template v-if="record.isFolder">
            <a-tag v-if="record.status === 'processing'" color="blue">
              <loading-outlined/>
              向量化中
            </a-tag>
            <a-tag v-else-if="record.status === 'ready'" color="green">完成</a-tag>
            <a-tag v-else-if="record.status === 'failed'" color="red">失败</a-tag>
          </template>
          <template v-else>
            <a-tag v-if="record.status === 'processing'" color="blue">
              <loading-outlined/>
              向量化中
            </a-tag>
            <a-tag v-else-if="record.status === 'ready'" color="green">完成</a-tag>
            <a-tooltip v-else-if="record.status === 'failed'" :title="record.errorMessage || '文档处理失败'">
              <a-tag color="red">失败</a-tag>
            </a-tooltip>
            <a-tag v-else color="default">{{ record.status }}</a-tag>
            <div v-if="record.status === 'failed' && record.errorMessage" class="document-error-text" :title="record.errorMessage">
              {{ record.errorMessage }}
            </div>
          </template>
        </template>
        <template v-if="column.key === 'action'">
          <template v-if="record.isFolder">
            <a-popconfirm
                title="确定要删除这个文件夹及其所有内容吗？"
                ok-text="确定"
                cancel-text="取消"
                @confirm="handleDeleteFolder(record.fileName)"
            >
              <a-button type="link" danger size="small">删除</a-button>
            </a-popconfirm>
          </template>
          <template v-else>
            <!-- Mobile View: Dropdown -->
            <a-dropdown v-if="isMobile" :trigger="['click']">
              <a-button type="text" size="small">
                <more-outlined/>
              </a-button>
              <template #overlay>
                <a-menu>
                  <a-menu-item @click="handlePreview(record)">预览</a-menu-item>
                  <a-menu-item @click="handleDownload(record)">下载</a-menu-item>
                  <a-menu-item @click="handlePreviewChunks(record)">预览切片</a-menu-item>
                  <a-menu-item v-if="record.status === 'failed'" @click="handleReprocess(record)">重试处理</a-menu-item>
                  <a-menu-item danger>
                    <a-popconfirm
                        title="确定要删除这个文件吗？"
                        ok-text="确定"
                        cancel-text="取消"
                        @confirm="handleDelete(record)"
                    >
                      <span>删除</span>
                    </a-popconfirm>
                  </a-menu-item>
                </a-menu>
              </template>
            </a-dropdown>

            <!-- Desktop View: Buttons -->
            <span v-else class="kb-desktop-actions">
                  <a-button type="link" size="small" @click="handlePreview(record)">预览</a-button>
                  <a-button type="link" size="small" @click="handlePreviewChunks(record)">切片</a-button>
                  <template v-if="record.status === 'failed'">
                    <a-button type="link" size="small" danger @click="handleReprocess(record)">重试</a-button>
                  </template>
                  
                  <a-dropdown>
                    <a class="ant-dropdown-link" @click.prevent style="font-size: 12px">
                      更多 <down-outlined/>
                    </a>
                    <template #overlay>
                      <a-menu>
                        <a-menu-item @click="handleDownload(record)">
                            下载文件
                        </a-menu-item>
                        <a-menu-item danger>
                            <a-popconfirm
                                title="确定要删除这个文件吗？"
                                ok-text="确定"
                                cancel-text="取消"
                                @confirm="handleDelete(record)"
                            >
                                <div style="width: 100%">删除文件</div>
                            </a-popconfirm>
                        </a-menu-item>
                      </a-menu>
                    </template>
                  </a-dropdown>
              </span>
          </template>
        </template>
      </template>
    </a-table>

    <!-- 设置对话框 -->
    <a-modal
        v-model:visible="settingsModalVisible"
        title="知识库设置"
        :confirm-loading="settingsSubmitting"
        @ok="handleSettingsSubmit"
        width="600px">
      <a-form :model="settingsForm" layout="vertical">
        <a-form-item label="知识库名称" required>
          <a-input v-model:value="settingsForm.name" placeholder="请输入知识库名称"/>
        </a-form-item>
        <a-form-item label="描述">
          <a-textarea v-model:value="settingsForm.description" placeholder="请输入描述" :rows="2"/>
        </a-form-item>
        <a-form-item label="系统提示词">
          <a-textarea v-model:value="settingsForm.systemPrompt" :placeholder="systemPromptPlaceholder" :rows="6"/>
        </a-form-item>
        <a-form-item label="可见性">
          <template #help>
            <span v-if="settingsForm.visibility === 'shared'">共享后，当前工作空间的所有成员均可访问此知识库。</span>
            <span v-else-if="settingsForm.visibility === 'public'">公开后，所有用户均可访问此知识库。</span>
            <span v-else>私有模式下，仅您本人可访问（被邀请用户除外）。</span>
          </template>
          <a-radio-group v-model:value="settingsForm.visibility" option-type="button">
            <a-radio value="private">私有</a-radio>
            <a-radio value="shared">工作空间共享</a-radio>
            <a-radio value="public">公开</a-radio>
          </a-radio-group>
        </a-form-item>
      </a-form>
    </a-modal>

    <a-modal
        v-model:visible="previewVisible"
        :title="previewTitle"
        :width="isMobile ? '100%' : 860"
        :footer="null"
        @cancel="handlePreviewCancel"
        :style="isMobile ? { top: 0, margin: 0, maxWidth: '100%' } : { top: '6vh' }"
        :body-style="isMobile ? { padding: '10px', height: 'calc(100vh - 55px)' } : {}"
    >
      <iframe v-if="previewType === 'pdf'" :src="previewContent" class="preview-pdf" title="PDF 预览"/>
      <div v-else-if="previewType === 'markdown'" class="markdown-body preview-scroll"
           v-html="md.render(previewContent)"></div>
      <div v-else-if="previewType === 'image'" class="preview-image-wrap">
        <img :src="previewContent" class="preview-image" alt="文件预览"/>
      </div>
      <DocxPreview v-else-if="previewType === 'docx' && docxBlob" :blob="docxBlob"/>
      <pre v-else class="preview-text">{{ previewContent }}</pre>
    </a-modal>

    <a-drawer
        v-model:visible="chunksDrawerVisible"
        title="切片预览"
        :width="isMobile ? '100%' : 600"
        @close="closeChunksDrawer">
      <a-list
          :loading="chunksLoading"
          item-layout="vertical"
          :data-source="chunksList"
      >
        <template #renderItem="{ item }">
          <a-list-item>
            <a-list-item-meta :title="`Chunk #${item.chunkIndex}`"/>
            <div style="white-space: pre-wrap; background: #f5f5f5; padding: 10px; border-radius: 4px;">{{
                item.text
              }}
            </div>
            <div style="margin-top: 8px; color: #999; font-size: 12px;">Token Length: {{ item.tokenLength }}</div>
          </a-list-item>
        </template>
        <template #loadMore>
          <div
              v-if="!chunksLoading && hasMoreChunks"
              :style="{ textAlign: 'center', marginTop: '12px', height: '32px', lineHeight: '32px' }"
          >
            <a-button @click="loadMoreChunks">加载更多</a-button>
          </div>
        </template>
      </a-list>
    </a-drawer>

    <!-- 邀请用户对话框 -->
    <a-modal
        v-model:visible="inviteModalVisible"
        title="邀请用户访问知识库"
        :confirm-loading="inviteSubmitting"
        @ok="handleInviteSubmit"
    >
      <a-form :model="inviteFormData" :label-col="{ span: 6 }" :wrapper-col="{ span: 18 }">
        <a-form-item label="用户名或邮箱" required>
          <a-input v-model:value="inviteFormData.userIdentifier" placeholder="请输入用户名或邮箱"/>
        </a-form-item>
        <a-alert
            message="只有私有知识库可以邀请用户。被邀请的用户将获得查看和使用该知识库的权限，但无法上传或修改文档"
            type="info"
            show-icon
            style="margin-top: 12px;"
        />
      </a-form>
    </a-modal>

    <!-- 被邀请用户列表对话框 -->
    <a-modal
        v-model:visible="invitedUsersModalVisible"
        title="被邀请用户列表"
        :footer="null"
        width="800px"
    >
      <a-table
          :columns="invitedUsersColumns"
          :data-source="invitedUsers"
          :loading="loadingInvitedUsers"
          :pagination="false"
          row-key="id"
      >
        <template #bodyCell="{ column, record }">
          <template v-if="column.key === 'grantedAt'">
            {{ new Date(record.grantedAt).toLocaleString() }}
          </template>
          <template v-if="column.key === 'action'">
            <a-button
                type="link"
                danger
                @click="handleRemoveInvitedUser(record)">
              移除
            </a-button>
          </template>
        </template>
      </a-table>
    </a-modal>

    <!-- 上传进度对话框 -->
    <a-modal
        v-model:visible="uploadProgressModalVisible"
        title="上传进度"
        :footer="null"
        :maskClosable="false"
        width="600px"
    >
      <!-- 整体进度统计 -->
      <div style="padding: 12px; background: #f5f5f5; border-radius: 4px; margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
          <span style="font-weight: 500;">整体进度</span>
          <span style="font-size: 16px; font-weight: 600; color: #1890ff;">
            {{ uploadProgressList.filter(item => item.status === 'done').length }} / {{ uploadProgressList.length }}
          </span>
        </div>
        <a-progress
            :percent="Math.round((uploadProgressList.filter(item => item.status === 'done').length / uploadProgressList.length) * 100)"
            :status="uploadProgressList.some(item => item.status === 'error') ? 'exception' : (uploadProgressList.length > 0 && uploadProgressList.every(item => item.status === 'done' && !item.processing) ? 'success' : 'active')"
            :show-info="false"
        />
        <div style="display: flex; gap: 16px; margin-top: 8px; font-size: 12px; color: #666;">
          <span>✅ 已上传: {{ uploadProgressList.filter(item => item.status === 'done').length }}</span>
          <span v-if="uploadProgressList.some(item => item.processing)">⏳ 处理中: {{ uploadProgressList.filter(item => item.processing).length }}</span>
          <span>⏳ 进行中: {{ uploadProgressList.filter(item => item.status === 'uploading').length }}</span>
          <span v-if="uploadProgressList.filter(item => item.status === 'error').length > 0" style="color: #ff4d4f;">
            ❌ 失败: {{ uploadProgressList.filter(item => item.status === 'error').length }}
          </span>
        </div>
      </div>

      <div style="max-height: 400px; overflow-y: auto;">
        <a-list :data-source="uploadProgressList" item-layout="horizontal">
          <template #renderItem="{ item }">
            <a-list-item>
              <a-list-item-meta :title="item.name">
                <template #description>
                  <a-progress :percent="item.percent"
                              :status="item.status === 'error' ? 'exception' : (item.status === 'done' && !item.processing ? 'success' : 'active')"/>
                  <div v-if="item.status === 'done' && item.processing" style="color: #1890ff">已上传，正在处理</div>
                  <div v-else-if="item.status === 'done'" style="color: #52c41a">处理完成</div>
                  <div v-if="item.status === 'error'" style="color: red">{{ item.error }}</div>
                </template>
              </a-list-item-meta>
            </a-list-item>
          </template>
        </a-list>
      </div>
      <div style="text-align: right; margin-top: 16px;">
        <a-button @click="uploadProgressModalVisible = false">关闭</a-button>
      </div>
    </a-modal>

    <!-- 下载/预览进度对话框 -->
    <a-modal
        v-model:visible="downloadProgress.visible"
        :title="downloadProgress.title"
        :footer="null"
        :closable="false"
        :maskClosable="false"
        width="400px"
        :centered="true"
    >
      <div style="padding: 24px 0; text-align: center;">
        <a-spin size="large"/>
        <div style="margin-top: 16px; color: #666;">
          正在处理，请稍候...
        </div>
      </div>
    </a-modal>
  </div>
</template>

<style scoped>
:deep(.markdown-body p) {
  margin-bottom: 0;
}

.markdown-body {
  line-height: 1.6;
}

.preview-pdf {
  width: 100%;
  height: 72vh;
  border: 0;
}

.preview-image-wrap {
  text-align: center;
}

.preview-image {
  max-width: 100%;
  max-height: 72vh;
}

.preview-scroll,
.preview-text {
  max-height: 72vh;
  margin: 0;
  overflow: auto;
}

.preview-text {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

/* ============ 页面骨架 ============ */
.kb-page {
  padding: 24px;
  height: 100vh;
  display: flex;
  flex-direction: column;
}

/* ============ 页头 ============ */
.kb-header {
  flex-shrink: 0;
  margin-bottom: 16px;
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
}

.kb-title-container {
  /* 保证标题区有足够宽度，空间不足时让操作按钮换行而不是把标题压扁 */
  flex: 1 1 260px;
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
}

.kb-back-btn {
  flex: none;
  color: v-bind('token.colorTextSecondary');
  transition: color 0.2s ease;
}

.kb-back-btn:hover {
  color: v-bind('token.colorPrimary');
}

.kb-title-wrap {
  flex: 1;
  min-width: 0;
}

.kb-title {
  margin: 0;
  font-size: 20px;
  font-weight: 600;
  line-height: 1.3;
  color: v-bind('token.colorText');
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.kb-meta {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 4px;
  font-size: 12px;
  color: v-bind('token.colorTextTertiary');
}

.kb-meta-dot {
  color: v-bind('token.colorTextQuaternary');
}

.kb-visibility-tag {
  margin-inline-end: 0;
}

.kb-actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
  flex: none;
}

/* ============ 工具栏 ============ */
.kb-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
  flex-shrink: 0;
  flex-wrap: wrap;
}

.kb-breadcrumb {
  min-width: 0;
  padding: 5px 12px;
  border-radius: 8px;
  background: v-bind('token.colorFillTertiary');
  overflow-x: auto;
  scrollbar-width: none;
}

.kb-breadcrumb::-webkit-scrollbar {
  display: none;
}

.kb-search-input {
  width: 280px;
  max-width: 100%;
}

.kb-toolbar-right {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-left: auto;
}

.kb-search-input :deep(.ant-input-prefix) {
  color: v-bind('token.colorTextTertiary');
}

.kb-processing-alert {
  margin-bottom: 12px;
  flex-shrink: 0;
  border-radius: 8px;
}

/* ============ 文件表格 ============ */
.kb-table {
  border: 1px solid v-bind('token.colorSplit');
  border-radius: 12px;
  overflow: hidden;
}

.kb-table :deep(.ant-table-thead > tr > th) {
  background: v-bind('token.colorFillTertiary');
  font-weight: 600;
  color: v-bind('token.colorText');
}

.kb-cell-icon {
  font-size: 16px;
  color: v-bind('token.colorTextTertiary');
}

.kb-cell-icon-folder {
  color: v-bind('token.colorPrimary');
}

.kb-file-name {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.kb-file-link {
  color: v-bind('token.colorText');
  transition: color 0.2s ease;
}

.kb-file-link-folder {
  font-weight: 600;
}

.kb-file-link:hover {
  color: v-bind('token.colorPrimary');
}

.document-error-text {
  max-width: 220px;
  margin-top: 4px;
  overflow: hidden;
  color: v-bind('token.colorError');
  font-size: 12px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.kb-table :deep(.ant-table-tbody > tr > td:last-child) {
  white-space: nowrap;
}

.kb-desktop-actions {
  display: inline-flex;
  align-items: center;
}

/* ============ 移动端文件卡片 ============ */
.kb-mobile-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.kb-mobile-empty {
  padding: 32px 0;
}

.kb-mobile-card {
  display: flex;
  align-items: flex-start;
  gap: 4px;
  padding: 12px;
  border: 1px solid v-bind('token.colorSplit');
  border-radius: 10px;
  background: v-bind('token.colorBgContainer');
  transition: border-color 0.2s ease, background-color 0.2s ease;
}

.kb-mobile-main {
  flex: 1;
  min-width: 0;
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 0;
  border: 0;
  background: transparent;
  color: inherit;
  font: inherit;
  text-align: left;
  appearance: none;
  cursor: pointer;
}

.kb-mobile-card:active {
  background: v-bind('token.colorFillQuaternary');
}

.kb-mobile-icon {
  flex: none;
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  background: v-bind('token.colorFillTertiary');
  color: v-bind('token.colorTextSecondary');
  font-size: 18px;
}

.kb-mobile-icon.is-folder {
  background: v-bind('token.colorPrimaryBg');
  color: v-bind('token.colorPrimary');
}

.kb-mobile-body {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.kb-mobile-name {
  overflow: hidden;
  color: v-bind('token.colorText');
  font-size: 14px;
  font-weight: 500;
  line-height: 1.4;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.kb-mobile-meta {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  color: v-bind('token.colorTextTertiary');
  font-size: 12px;
  line-height: 1.5;
}

.kb-mobile-status {
  margin-inline-end: 0;
  padding-inline: 6px;
  font-size: 11px;
  line-height: 18px;
}

.kb-mobile-error {
  overflow: hidden;
  color: v-bind('token.colorError');
  font-size: 12px;
  line-height: 1.4;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.kb-mobile-more {
  flex: none;
  margin-top: -2px;
  color: v-bind('token.colorTextTertiary');
}

@media (max-width: 768px) {
  .preview-pdf,
  .preview-scroll,
  .preview-text {
    height: calc(100vh - 80px);
    max-height: none;
  }

  .preview-image {
    max-height: calc(100vh - 80px);
  }

  .kb-page {
    height: auto;
    min-height: 100vh;
    min-height: 100dvh;
    padding: 16px;
    padding-bottom: calc(16px + env(safe-area-inset-bottom));
  }

  .kb-header {
    flex-direction: column;
    align-items: stretch;
    gap: 12px;
  }

  .kb-title-container {
    flex: none;
    width: 100%;
  }

  .kb-title {
    font-size: 18px;
  }

  .kb-actions {
    width: 100%;
    justify-content: space-between;
    flex-wrap: wrap;
    /* 给左上角的悬浮菜单按钮留出位置，避免遮挡第一个操作按钮 */
    padding-left: 40px;
  }

  .kb-toolbar-right {
    width: 100%;
  }

  .kb-search-input {
    flex: 1;
    width: auto;
  }

  .kb-actions .ant-btn {
    flex: 1;
    min-width: 40px;
    padding: 4px 8px;
    display: flex;
    justify-content: center;
    align-items: center;
  }

  .kb-actions .ant-upload-wrapper {
    flex: 1;
  }

  .kb-actions .ant-upload-wrapper .ant-btn {
    width: 100%;
  }
}
</style>
