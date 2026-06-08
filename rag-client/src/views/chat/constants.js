import {
  GlobalOutlined,
  FileSearchOutlined,
  CodeOutlined,
} from '@ant-design/icons-vue'

export const allKnownTools = ['webSearch']

export const toolConfigs = {
  'webSearch': {icon: GlobalOutlined, label: '联网搜索', desc: '开启联网搜索'},
  'web_extractor': {icon: FileSearchOutlined, label: '网页', desc: '网页提取'},
  'code_interpreter': {icon: CodeOutlined, label: '代码', desc: '代码解释器'},
}

export const assistantAvatar = {
  color: '#f56a00',
  backgroundColor: '#fde3cf',
}

export const userAvatar = {
  color: '#fff',
  backgroundColor: '#87d068',
}
