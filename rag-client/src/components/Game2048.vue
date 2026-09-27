<script setup>
import {computed, onMounted, onUnmounted} from 'vue'
import {useGame2048Store} from '@/stores/game2048'
import {ReloadOutlined, UndoOutlined, ThunderboltOutlined} from '@ant-design/icons-vue'

const game = useGame2048Store()

const KEY_DIRS = {
  arrowup: 'U', w: 'U',
  arrowdown: 'D', s: 'D',
  arrowleft: 'L', a: 'L',
  arrowright: 'R', d: 'R',
}

const statusText = computed(() => {
  if (game.gameOver) return `游戏结束! 最大方块 ${game.maxTile}, 点「新游戏」再来一局`
  if (game.aiEnabled) return 'AI 托管中, 它正在替你冲击 2048...'
  if (game.won) return '达成 2048! 可以继续冲更高分'
  return '方向键 / WASD 移动, 手机滑动屏幕'
})

function onKeydown(e) {
  const tag = e.target?.tagName
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return
  const dir = KEY_DIRS[e.key.toLowerCase()]
  if (!dir) return
  e.preventDefault()
  if (!game.aiEnabled) game.move(dir)
}

// 移动端: 棋盘上滑动手势
let touchStart = null
function onTouchStart(e) {
  const t = e.touches[0]
  touchStart = {x: t.clientX, y: t.clientY}
}
function onTouchEnd(e) {
  if (!touchStart) return
  const t = e.changedTouches[0]
  const dx = t.clientX - touchStart.x
  const dy = t.clientY - touchStart.y
  touchStart = null
  const absX = Math.abs(dx)
  const absY = Math.abs(dy)
  if (Math.max(absX, absY) < 24) return
  const dir = absX > absY ? (dx > 0 ? 'R' : 'L') : (dy > 0 ? 'D' : 'U')
  if (!game.aiEnabled) game.move(dir)
}

onMounted(() => window.addEventListener('keydown', onKeydown))
onUnmounted(() => window.removeEventListener('keydown', onKeydown))
</script>

<template>
  <div class="g2048">
    <div class="g2048-header">
      <div class="g2048-title">
        2048
        <span class="g2048-badge">隐藏彩蛋</span>
      </div>
      <div class="g2048-scores">
        <div class="g2048-score-box">
          <span>分数</span>
          <strong>{{ game.score }}</strong>
        </div>
        <div class="g2048-score-box">
          <span>最高</span>
          <strong>{{ game.best }}</strong>
        </div>
        <div class="g2048-score-box">
          <span>最大</span>
          <strong>{{ game.maxTile }}</strong>
        </div>
      </div>
    </div>

    <div class="g2048-board" @touchstart="onTouchStart" @touchend="onTouchEnd">
      <!-- 背景格子 -->
      <div class="g2048-grid">
        <div v-for="i in 16" :key="i" class="g2048-grid-cell">
          <div class="g2048-grid-face"></div>
        </div>
      </div>
      <!-- 方块层: transform 平滑滑动; span 按值重建, 仅合并/新生时弹跳, 快速连续操作动画互不干扰 -->
      <div class="g2048-tiles">
        <div
          v-for="t in game.tiles"
          :key="t.id"
          class="g2048-tile"
          :class="{'is-dying': t.dying}"
          :style="{'--c': t.c, '--r': t.r}"
        >
          <div class="g2048-tile-face" :class="`tile-${Math.min(t.value, 4096)}`">
            <span v-if="!t.dying" :key="t.value" class="g2048-tile-value">{{ t.value }}</span>
          </div>
        </div>
      </div>
    </div>

    <div class="g2048-status">{{ statusText }}</div>

    <div class="g2048-controls">
      <a-button type="primary" @click="game.newGame()">
        <template #icon>
          <reload-outlined />
        </template>
        新游戏
      </a-button>
      <a-button :disabled="!game.history.length" @click="game.undo()">
        <template #icon>
          <undo-outlined />
        </template>
        撤销
      </a-button>
      <div class="g2048-ai-toggle" :class="{ 'ai-on': game.aiEnabled }">
        <thunderbolt-outlined />
        <span class="g2048-ai-label">AI 托管</span>
        <a-switch
          size="small"
          :checked="game.aiEnabled"
          checked-children="开"
          un-checked-children="关"
          @change="game.setAiEnabled"
        />
      </div>
    </div>
  </div>
</template>

<style scoped>
.g2048 {
  display: flex;
  flex-direction: column;
  gap: 12px;
  width: min(100%, 400px);
  margin: 0 auto;
  user-select: none;
}

.g2048-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  padding-right: 40px; /* 给弹窗右上角关闭按钮留位, 避免与分数栏重叠 */
}

.g2048-title {
  font-size: 22px;
  font-weight: 700;
  color: #776e65;
  display: flex;
  align-items: center;
  gap: 8px;
}

[data-theme='dark'] .g2048-title {
  color: #e8e6e3;
}

.g2048-badge {
  font-size: 11px;
  font-weight: 500;
  padding: 2px 8px;
  border-radius: 10px;
  color: #1677ff;
  background: rgba(22, 119, 255, 0.12);
}

.g2048-scores {
  display: flex;
  gap: 8px;
}

.g2048-score-box {
  min-width: 56px;
  padding: 4px 10px;
  border-radius: 8px;
  text-align: center;
  background: #bbada0;
  color: #f7f4f0;
  display: flex;
  flex-direction: column;
  line-height: 1.3;
}

.g2048-score-box span {
  font-size: 11px;
  opacity: 0.85;
}

.g2048-score-box strong {
  font-size: 16px;
}

.g2048-board {
  position: relative;
  aspect-ratio: 1;
  border-radius: 12px;
  background: #bbada0;
  touch-action: none; /* 移动端滑动时避免页面滚动 */
}

.g2048-grid {
  position: absolute;
  inset: 0;
  display: grid;
  grid-template-columns: repeat(4, 1fr);
}

.g2048-grid-cell {
  aspect-ratio: 1;
  padding: 4px;
  box-sizing: border-box;
}

.g2048-grid-face {
  width: 100%;
  height: 100%;
  border-radius: 8px;
  background: #cdc1b4;
}

.g2048-tiles {
  position: absolute;
  inset: 0;
}

.g2048-tile {
  position: absolute;
  width: 25%;
  height: 25%;
  box-sizing: border-box;
  padding: 4px;
  transition: transform 0.12s ease;
  transform: translate(calc(var(--c) * 100%), calc(var(--r) * 100%));
}

.g2048-tile.is-dying {
  z-index: 1;
}

.g2048-tile.is-dying .g2048-tile-face {
  opacity: 0;
  transition: opacity 0.12s ease;
}

.g2048-tile-face {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  font-size: clamp(18px, 5.5vw, 28px);
  font-weight: 700;
}

.g2048-tile-value {
  animation: g2048-pop 0.12s ease-out;
}

@media (prefers-reduced-motion: reduce) {
  .g2048-tile {
    transition: none;
  }

  .g2048-tile-value {
    animation: none;
  }
}

@keyframes g2048-pop {
  from {
    transform: scale(0.86);
    opacity: 0.6;
  }
  to {
    transform: scale(1);
    opacity: 1;
  }
}

.tile-2 { background: #eee4da; color: #776e65; }
.tile-4 { background: #ede0c8; color: #776e65; }
.tile-8 { background: #f2b179; color: #f9f6f2; }
.tile-16 { background: #f59563; color: #f9f6f2; }
.tile-32 { background: #f67c5f; color: #f9f6f2; }
.tile-64 { background: #f65e3b; color: #f9f6f2; }
.tile-128 { background: #edcf72; color: #f9f6f2; }
.tile-256 { background: #edcc61; color: #f9f6f2; }
.tile-512 { background: #edc850; color: #f9f6f2; }
.tile-1024 { background: #edc53f; color: #f9f6f2; }
.tile-2048 { background: #edc22e; color: #f9f6f2; }
.tile-4096 { background: #3c3a32; color: #f9f6f2; }

.g2048-status {
  min-height: 20px;
  font-size: 13px;
  color: #8c8c8c;
  text-align: center;
}

[data-theme='dark'] .g2048-status {
  color: #999;
}

.g2048-controls {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.g2048-ai-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-left: auto;
  padding: 4px 10px;
  border-radius: 8px;
  border: 1px solid rgba(0, 0, 0, 0.1);
  transition: all 0.2s;
}

[data-theme='dark'] .g2048-ai-toggle {
  border-color: rgba(255, 255, 255, 0.14);
}

.g2048-ai-toggle.ai-on {
  color: #faad14;
  border-color: rgba(250, 173, 20, 0.5);
  background: rgba(250, 173, 20, 0.08);
}

.g2048-ai-label {
  font-size: 13px;
}

/* 移动端: 更紧凑的间距与更大的触控目标 */
@media (max-width: 768px) {
  .g2048 {
    gap: 10px;
  }

  .g2048-board {
    border-radius: 10px;
  }

  .g2048-grid-cell,
  .g2048-tile {
    padding: 3px;
  }

  .g2048-controls :deep(.ant-btn) {
    flex: 1;
    min-height: 38px;
  }

  .g2048-ai-toggle {
    width: 100%;
    margin-left: 0;
    justify-content: center;
    min-height: 38px;
  }
}
</style>
