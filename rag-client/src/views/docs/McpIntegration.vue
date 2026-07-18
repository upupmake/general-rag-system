<script setup>
import {computed, onMounted, onUnmounted, ref} from 'vue'
import {useRouter} from 'vue-router'
import {ArrowUpOutlined, MenuOutlined} from '@ant-design/icons-vue'
import md from '@/utils/markdown.js'
import source from '@/docs/mcp-integration.md?raw'
import logo from '@/assets/logo.png'
import image1 from '@/docs/images/1.png'
import image2 from '@/docs/images/2.png'
import image3 from '@/docs/images/3.png'
import image4 from '@/docs/images/4.png'
import image5 from '@/docs/images/5.png'
import image6 from '@/docs/images/6.png'
import image7 from '@/docs/images/7.png'
import image8 from '@/docs/images/8.png'
import image9 from '@/docs/images/9.png'
import image10 from '@/docs/images/10.png'
import image11 from '@/docs/images/11.png'
import image12 from '@/docs/images/12.png'

const router = useRouter()
const tocOpen = ref(false)
const showBackTop = ref(false)
const activeSection = ref('before-start')

const sections = [
  {id: 'before-start', label: '开始之前'},
  {id: 'install-agent', label: '1. 安装 Agent'},
  {id: 'access-key', label: '2. 创建 Access Key'},
  {id: 'mcp-config', label: '3. 添加 MCP'},
  {id: 'model-config', label: '4. 配置模型'},
  {id: 'first-search', label: '5. 开始检索'},
  {id: 'tools', label: '8 个可用工具'},
  {id: 'file-management', label: '管理私有文件'},
  {id: 'troubleshooting', label: '常见问题排查'},
  {id: 'security', label: '安全建议'}
]

const images = [image1, image2, image3, image4, image5, image6, image7, image8, image9, image10, image11, image12]
const contentWithImages = images.reduce(
  (content, image, index) => content.replaceAll(`./images/${index + 1}.png`, image),
  source
)
const renderedContent = computed(() => md.render(contentWithImages))

const scrollToSection = id => {
  document.getElementById(id)?.scrollIntoView({behavior: 'smooth', block: 'start'})
  tocOpen.value = false
}

const scrollToTop = () => window.scrollTo({top: 0, behavior: 'smooth'})

const handleDocumentClick = event => {
  const link = event.target.closest('.mcp-markdown a')
  if (!link) return

  const href = link.getAttribute('href')
  if (href?.startsWith('/') && !link.hasAttribute('target')) {
    event.preventDefault()
    router.push(href)
  }
}

const handleScroll = () => {
  showBackTop.value = window.scrollY > 560
  let current = sections[0].id
  for (const section of sections) {
    const element = document.getElementById(section.id)
    if (element && element.getBoundingClientRect().top <= 150) {
      current = section.id
    }
  }
  activeSection.value = current
}

onMounted(() => {
  document.title = 'MCP 接入指南 - RAG知识系统'
  window.addEventListener('scroll', handleScroll, {passive: true})
  handleScroll()
})

onUnmounted(() => window.removeEventListener('scroll', handleScroll))
</script>

<template>
  <div class="docs-page" @click="handleDocumentClick">
    <header class="docs-header">
      <router-link class="brand" to="/">
        <img :src="logo" alt="General RAG" />
        <span>General RAG</span>
        <i>DOCS</i>
      </router-link>
      <nav class="header-actions" aria-label="文档操作">
        <a href="https://makecode.forwardforever.top/" target="_blank" rel="noopener noreferrer">下载 Agent</a>
        <router-link class="console-link" to="/access-keys">进入控制台</router-link>
      </nav>
      <button class="mobile-toc-button" type="button" aria-label="打开文档目录" @click="tocOpen = !tocOpen">
        <menu-outlined />
      </button>
    </header>

    <div class="docs-shell">
      <aside class="docs-toc" :class="{open: tocOpen}">
        <div class="toc-title">接入指南</div>
        <button
          v-for="section in sections"
          :key="section.id"
          type="button"
          :class="{active: activeSection === section.id}"
          @click="scrollToSection(section.id)"
        >
          {{ section.label }}
        </button>
        <div class="toc-meta">
          <span>协议</span>
          <strong>MCP · Streamable HTTP</strong>
          <span>文档状态</span>
          <strong><i /> 服务可用</strong>
        </div>
      </aside>

      <main class="docs-content">
        <article class="mcp-markdown markdown-body" v-html="renderedContent" />
        <footer class="docs-footer">
          <span>General RAG MCP</span>
          <span>请勿在文档、日志或截图中暴露完整 Access Key</span>
        </footer>
      </main>
    </div>

    <button v-show="showBackTop" class="back-top" type="button" aria-label="返回顶部" @click="scrollToTop">
      <arrow-up-outlined />
    </button>
  </div>
</template>

<style scoped>
.docs-page {
  --ink: #17221c;
  --muted: #607068;
  --line: #dce3de;
  --paper: #fbfcf9;
  --forest: #174c37;
  --acid: #b8ee53;
  min-height: 100vh;
  color: var(--ink);
  background:
    linear-gradient(rgba(23, 76, 55, 0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(23, 76, 55, 0.035) 1px, transparent 1px),
    var(--paper);
  background-size: 32px 32px;
  font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
}

.docs-header {
  position: sticky;
  z-index: 20;
  top: 0;
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 36px;
  border-bottom: 1px solid rgba(23, 76, 55, 0.16);
  background: rgba(251, 252, 249, 0.92);
  backdrop-filter: blur(16px);
}

.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  color: var(--ink);
  font-family: Georgia, 'Microsoft YaHei', serif;
  font-size: 18px;
  font-weight: 700;
  text-decoration: none;
}

.brand img {
  width: 30px;
  height: 30px;
  object-fit: contain;
}

.brand i {
  padding: 3px 6px;
  color: var(--forest);
  border: 1px solid var(--forest);
  border-radius: 3px;
  font-family: Consolas, monospace;
  font-size: 9px;
  font-style: normal;
  letter-spacing: 1px;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 22px;
  font-size: 13px;
  font-weight: 600;
}

.header-actions a {
  color: var(--ink);
  text-decoration: none;
}

.header-actions .console-link {
  padding: 8px 14px;
  color: #f7fff0;
  border-radius: 4px;
  background: var(--forest);
}

.mobile-toc-button {
  display: none;
  width: 38px;
  height: 38px;
  border: 1px solid var(--line);
  border-radius: 4px;
  color: var(--ink);
  background: transparent;
}

.docs-shell {
  width: min(1260px, 100%);
  margin: 0 auto;
  display: grid;
  grid-template-columns: 230px minmax(0, 820px);
  gap: 72px;
  justify-content: center;
}

.docs-toc {
  position: sticky;
  top: 64px;
  height: calc(100vh - 64px);
  padding: 44px 0 28px;
  border-right: 1px solid var(--line);
}

.toc-title {
  margin-bottom: 18px;
  color: var(--muted);
  font-family: Consolas, monospace;
  font-size: 11px;
  letter-spacing: 1.4px;
  text-transform: uppercase;
}

.docs-toc button {
  width: calc(100% - 28px);
  display: block;
  padding: 8px 12px;
  border: 0;
  border-left: 2px solid transparent;
  color: var(--muted);
  background: transparent;
  cursor: pointer;
  font-size: 13px;
  text-align: left;
  transition: color 0.15s, border-color 0.15s, background 0.15s;
}

.docs-toc button:hover,
.docs-toc button.active {
  color: var(--forest);
  border-left-color: var(--acid);
  background: linear-gradient(90deg, rgba(184, 238, 83, 0.14), transparent);
  font-weight: 700;
}

.toc-meta {
  position: absolute;
  right: 28px;
  bottom: 28px;
  left: 0;
  display: grid;
  gap: 4px;
  padding-top: 18px;
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: 10px;
}

.toc-meta strong {
  margin-bottom: 8px;
  color: var(--ink);
  font-family: Consolas, monospace;
  font-size: 10px;
}

.toc-meta strong i {
  display: inline-block;
  width: 6px;
  height: 6px;
  margin-right: 4px;
  border-radius: 50%;
  background: #35a55b;
}

.docs-content {
  min-width: 0;
  padding: 72px 0 48px;
}

.mcp-markdown {
  color: var(--ink);
  background: transparent;
  font-size: 15px;
  line-height: 1.85;
}

.mcp-markdown :deep(.doc-kicker) {
  margin-bottom: 18px;
  color: var(--forest);
  font-family: Consolas, monospace;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 2px;
}

.mcp-markdown :deep(h1) {
  max-width: 680px;
  margin: 0 0 20px;
  padding: 0;
  border: 0;
  color: var(--ink);
  font-family: Georgia, 'Noto Serif SC', 'Microsoft YaHei', serif;
  font-size: clamp(40px, 6vw, 68px);
  font-weight: 500;
  line-height: 1.08;
  letter-spacing: -2px;
}

.mcp-markdown :deep(h1 + p) {
  max-width: 670px;
  margin-bottom: 24px;
  color: var(--muted);
  font-size: 18px;
  line-height: 1.8;
}

.mcp-markdown :deep(h2) {
  scroll-margin-top: 96px;
  margin: 76px 0 22px;
  padding: 0 0 12px;
  border-bottom: 1px solid var(--line);
  color: var(--ink);
  font-family: Georgia, 'Noto Serif SC', 'Microsoft YaHei', serif;
  font-size: 30px;
  font-weight: 600;
  letter-spacing: -0.5px;
}

.mcp-markdown :deep(h3) {
  margin: 40px 0 14px;
  color: var(--forest);
  font-size: 19px;
}

.mcp-markdown :deep(p),
.mcp-markdown :deep(li) {
  color: #34433b;
}

.mcp-markdown :deep(a) {
  color: #146747;
  text-decoration-color: rgba(20, 103, 71, 0.35);
  text-underline-offset: 3px;
}

.mcp-markdown :deep(blockquote) {
  margin: 24px 0;
  padding: 14px 18px;
  color: #34433b;
  border-left: 3px solid var(--acid);
  background: rgba(184, 238, 83, 0.11);
}

.mcp-markdown :deep(pre) {
  position: relative;
  margin: 18px 0 26px;
  padding: 20px;
  overflow: auto;
  border: 1px solid #244e3b;
  border-radius: 5px;
  background: #102b20 !important;
  box-shadow: 7px 7px 0 rgba(23, 76, 55, 0.1);
}

.mcp-markdown :deep(pre code) {
  color: #e9f6eb;
  background: transparent;
  font-family: Consolas, 'Courier New', monospace;
  font-size: 13px;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.mcp-markdown :deep(code) {
  padding: 2px 5px;
  color: #175e42;
  border-radius: 3px;
  background: #e9f0eb;
  font-family: Consolas, 'Courier New', monospace;
}

.mcp-markdown :deep(table) {
  display: table;
  width: 100%;
  margin: 24px 0 34px;
  border-collapse: collapse;
  overflow: hidden;
  font-size: 13px;
}

.mcp-markdown :deep(th) {
  padding: 11px 12px;
  color: #effbe9;
  border: 1px solid #315b48;
  background: var(--forest);
  font-family: Consolas, monospace;
  font-weight: 600;
  text-align: left;
}

.mcp-markdown :deep(td) {
  padding: 12px;
  border: 1px solid var(--line);
  vertical-align: top;
}

.mcp-markdown :deep(tr:nth-child(even) td) {
  background: rgba(23, 76, 55, 0.025);
}

.mcp-markdown :deep(.doc-lead-actions) {
  display: flex;
  gap: 12px;
  margin: 28px 0 34px;
}

.mcp-markdown :deep(.doc-lead-actions a) {
  padding: 10px 17px;
  border-radius: 4px;
  font-size: 13px;
  font-weight: 700;
  text-decoration: none;
}

.mcp-markdown :deep(.doc-primary-link) {
  color: white;
  background: var(--forest);
}

.mcp-markdown :deep(.doc-secondary-link) {
  color: var(--forest);
  border: 1px solid var(--forest);
}

.mcp-markdown :deep(.doc-fact-grid) {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  margin: 34px 0;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.62);
}

.mcp-markdown :deep(.doc-fact-grid > div) {
  padding: 22px;
  border-right: 1px solid var(--line);
}

.mcp-markdown :deep(.doc-fact-grid > div:last-child) {
  border-right: 0;
}

.mcp-markdown :deep(.doc-fact-grid strong),
.mcp-markdown :deep(.doc-fact-grid span) {
  display: block;
}

.mcp-markdown :deep(.doc-fact-grid strong) {
  color: var(--forest);
  font-family: Georgia, serif;
  font-size: 28px;
}

.mcp-markdown :deep(.doc-fact-grid span) {
  margin-top: 3px;
  color: var(--muted);
  font-size: 12px;
}

.mcp-markdown :deep(.doc-shot) {
  margin: 28px 0 38px;
  overflow: hidden;
  border: 1px solid #cdd8d1;
  border-radius: 6px;
  background: #f3f6f2;
  box-shadow: 9px 9px 0 rgba(23, 76, 55, 0.09);
}

.mcp-markdown :deep(.doc-shot img) {
  width: 100%;
  height: auto;
  display: block;
  border: 0;
  border-radius: 0;
  background: #141816;
}

.mcp-markdown :deep(.doc-shot figcaption) {
  display: grid;
  grid-template-columns: minmax(170px, 0.38fr) 1fr;
  gap: 18px;
  padding: 15px 18px;
  border-top: 1px solid #cdd8d1;
  background: rgba(255, 255, 255, 0.76);
}

.mcp-markdown :deep(.doc-shot figcaption strong) {
  color: var(--forest);
  font-family: Consolas, 'Microsoft YaHei', monospace;
  font-size: 12px;
}

.mcp-markdown :deep(.doc-shot figcaption span) {
  color: var(--muted);
  font-size: 12px;
  line-height: 1.65;
}

.mcp-markdown :deep(.doc-callout) {
  display: grid;
  grid-template-columns: 180px 1fr;
  gap: 20px;
  margin: 24px 0;
  padding: 18px 20px;
  border: 1px solid #a8d78e;
  background: #f1fae9;
}

.mcp-markdown :deep(.doc-callout strong) {
  color: var(--forest);
}

.mcp-markdown :deep(.doc-callout span) {
  color: #3f5147;
}

.mcp-markdown :deep(.doc-compare) {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1px;
  margin: 22px 0;
  border: 1px solid var(--line);
  background: var(--line);
}

.mcp-markdown :deep(.doc-compare > div) {
  padding: 24px;
  background: var(--paper);
}

.mcp-markdown :deep(.doc-compare h3) {
  margin: 8px 0;
}

.mcp-markdown :deep(.doc-compare p) {
  margin: 0;
  font-size: 13px;
}

.mcp-markdown :deep(.compare-label) {
  color: var(--muted);
  font-family: Consolas, monospace;
  font-size: 10px;
  letter-spacing: 1px;
}

.mcp-markdown :deep(.doc-finish) {
  margin-top: 80px;
  padding: 42px;
  color: #eef9e7;
  background: var(--forest);
  box-shadow: 10px 10px 0 var(--acid);
}

.mcp-markdown :deep(.doc-finish span) {
  color: var(--acid);
  font-family: Consolas, monospace;
  font-size: 11px;
  letter-spacing: 1px;
}

.mcp-markdown :deep(.doc-finish h2) {
  margin: 8px 0 22px;
  padding: 0;
  color: white;
  border: 0;
  font-size: 28px;
}

.mcp-markdown :deep(.doc-finish a) {
  display: inline-block;
  padding: 9px 14px;
  color: var(--forest);
  border-radius: 3px;
  background: var(--acid);
  font-weight: 700;
  text-decoration: none;
}

.docs-footer {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  margin-top: 72px;
  padding-top: 20px;
  color: var(--muted);
  border-top: 1px solid var(--line);
  font-family: Consolas, monospace;
  font-size: 10px;
}

.back-top {
  position: fixed;
  right: 24px;
  bottom: 24px;
  width: 42px;
  height: 42px;
  border: 0;
  border-radius: 4px;
  color: white;
  background: var(--forest);
  box-shadow: 4px 4px 0 var(--acid);
  cursor: pointer;
}

@media (max-width: 960px) {
  .docs-shell {
    display: block;
  }

  .docs-content {
    width: min(760px, calc(100% - 40px));
    margin: 0 auto;
    padding-top: 52px;
  }

  .docs-toc {
    position: fixed;
    z-index: 30;
    top: 64px;
    right: 0;
    bottom: 0;
    left: auto;
    width: min(300px, 82vw);
    height: auto;
    padding: 26px 20px;
    border-right: 0;
    border-left: 1px solid var(--line);
    background: var(--paper);
    box-shadow: -20px 20px 50px rgba(17, 43, 31, 0.14);
    transform: translateX(105%);
    transition: transform 0.2s ease;
  }

  .docs-toc.open {
    transform: translateX(0);
  }

  .toc-meta {
    display: none;
  }

  .mobile-toc-button {
    display: inline-grid;
    place-items: center;
  }
}

@media (max-width: 640px) {
  .docs-header {
    height: 58px;
    padding: 0 16px;
  }

  .brand span {
    font-size: 15px;
  }

  .header-actions {
    display: none;
  }

  .docs-toc {
    top: 58px;
  }

  .docs-content {
    width: calc(100% - 28px);
    padding-top: 38px;
  }

  .mcp-markdown :deep(h1) {
    font-size: 39px;
    letter-spacing: -1.2px;
  }

  .mcp-markdown :deep(h1 + p) {
    font-size: 16px;
  }

  .mcp-markdown :deep(h2) {
    margin-top: 58px;
    font-size: 25px;
  }

  .mcp-markdown :deep(.doc-lead-actions) {
    display: grid;
  }

  .mcp-markdown :deep(.doc-lead-actions a) {
    text-align: center;
  }

  .mcp-markdown :deep(.doc-fact-grid) {
    grid-template-columns: 1fr;
  }

  .mcp-markdown :deep(.doc-fact-grid > div) {
    padding: 14px 18px;
    border-right: 0;
    border-bottom: 1px solid var(--line);
  }

  .mcp-markdown :deep(.doc-fact-grid > div:last-child) {
    border-bottom: 0;
  }

  .mcp-markdown :deep(.doc-shot) {
    margin: 22px 0 30px;
    box-shadow: 5px 5px 0 rgba(23, 76, 55, 0.09);
  }

  .mcp-markdown :deep(.doc-shot figcaption) {
    grid-template-columns: 1fr;
    gap: 3px;
    padding: 12px 14px;
  }

  .mcp-markdown :deep(.doc-callout),
  .mcp-markdown :deep(.doc-compare) {
    grid-template-columns: 1fr;
  }

  .mcp-markdown :deep(table) {
    display: block;
    overflow-x: auto;
    white-space: nowrap;
  }

  .mcp-markdown :deep(.doc-finish) {
    padding: 28px 22px;
    box-shadow: 6px 6px 0 var(--acid);
  }

  .docs-footer {
    flex-direction: column;
  }
}
</style>
