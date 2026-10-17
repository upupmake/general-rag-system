import markdownit from 'markdown-it'
import taskLists from 'markdown-it-task-lists'
import markdownMath from '@/utils/markdownMath.js'
import hljs from '@/utils/hljs.js'
import 'highlight.js/styles/atom-one-light.css'

const langAliases = {
  'js': 'javascript',
  'ts': 'typescript',
  'py': 'python',
  'h5': 'html',
  'rb': 'ruby',
  'sh': 'bash',
  'c++': 'cpp',
}

export const md = markdownit({
  html: false,
  linkify: true,
  typographer: true,
  highlight: function (str, lang) {
    lang = langAliases[lang] || lang
    if (lang && hljs.getLanguage(lang)) {
      try {
        return '<pre class="hljs"><code>'
            + hljs.highlight(str, {language: lang, ignoreIllegals: true}).value +
            '</code></pre>'
      } catch (__) {
      }
    }
    const result = hljs.highlightAuto(str)
    return '<pre class="hljs"><code>' + result.value + '</code></pre>'
  }
}).use(taskLists).use(markdownMath)

const renderCache = new WeakMap()

// 按消息对象缓存渲染结果(text -> html)，流式期间未变化的历史消息直接复用，避免重复 md.render/MathJax
export function renderCached(obj, slot, text) {
  let entry = renderCache.get(obj)
  if (!entry) {
    entry = {}
    renderCache.set(obj, entry)
  }
  const src = text || ''
  const cached = entry[slot]
  if (cached && cached.text === src) return cached.html
  const html = md.render(src)
  entry[slot] = {text: src, html}
  return html
}
