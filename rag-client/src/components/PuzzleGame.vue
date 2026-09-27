<script setup>
import {computed, onMounted, onUnmounted} from 'vue'
import {ReloadOutlined, UndoOutlined, EyeOutlined, ThunderboltOutlined} from '@ant-design/icons-vue'
import {useGamePuzzleStore} from '@/stores/gamePuzzle'
import puzzleImg from '@/assets/puzzle-image.jpg'

const game = useGamePuzzleStore()

const statusText = computed(() => {
  if (game.solved) return `已复原! 共 ${game.moves} 步`
  if (game.thinking) return 'AI 思考中…'
  if (game.aiEnabled) return 'AI 托管中…'
  return '点击空白格旁的方块进行移动(方向键亦可)'
})

// 方块显示图片切片: 编号 v 的方块显示其目标位置对应的那块图
function tileFaceStyle(value) {
  const n = game.n
  const home = value - 1
  const hr = Math.floor(home / n)
  const hc = home % n
  return {
    backgroundImage: `url(${puzzleImg})`,
    backgroundSize: `${n * 100}% ${n * 100}%`,
    backgroundPosition: n > 1 ? `${(hc * 100) / (n - 1)}% ${(hr * 100) / (n - 1)}%` : 'center',
  }
}

function onTileClick(t) {
  game.moveTile(t.value)
}

function onBoardClick() {
  if (game.view === 'goal') game.toggleView() // 点击目标图切回当前状态
}

function onSizeChange(e) {
  game.setSize(e.target.value)
}

function onKeydown(e) {
  const dirs = {
    ArrowUp: [-1, 0],
    ArrowDown: [1, 0],
    ArrowLeft: [0, -1],
    ArrowRight: [0, 1],
  }
  const d = dirs[e.key]
  if (!d || game.view !== 'game') return
  e.preventDefault()
  game.moveBlank(d[0], d[1])
}

let touchStart = null

function onTouchStart(e) {
  touchStart = e.touches[0]
}

function onTouchEnd(e) {
  if (!touchStart || game.view !== 'game') {
    touchStart = null
    return
  }
  const dx = e.changedTouches[0].clientX - touchStart.x
  const dy = e.changedTouches[0].clientY - touchStart.y
  touchStart = null
  if (Math.max(Math.abs(dx), Math.abs(dy)) < 24) return // 触点按点击处理
  if (Math.abs(dx) > Math.abs(dy)) game.moveBlank(0, dx > 0 ? 1 : -1)
  else game.moveBlank(dy > 0 ? 1 : -1, 0)
}

onMounted(() => window.addEventListener('keydown', onKeydown))
onUnmounted(() => window.removeEventListener('keydown', onKeydown))
</script>

<template>
  <div class="puzzle">
    <div class="puzzle-header">
      <div class="puzzle-title">
        图片拼图
        <span class="puzzle-badge">隐藏彩蛋</span>
      </div>
      <div class="puzzle-scores">
        <div class="puzzle-score-box">
          <span>步数</span>
          <strong>{{ game.moves }}</strong>
        </div>
        <div class="puzzle-score-box">
          <span>尺寸</span>
          <strong>{{ game.n }}x{{ game.n }}</strong>
        </div>
      </div>
    </div>

    <div
      class="puzzle-board"
      :class="{'is-goal': game.view === 'goal', 'is-solved': game.solved}"
      :style="{'--n': game.n}"
      @click="onBoardClick"
      @touchstart="onTouchStart"
      @touchend="onTouchEnd"
    >
      <!-- 目标图: 点击任意处切回当前状态 -->
      <div v-if="game.view === 'goal'" class="puzzle-goal" :style="{backgroundImage: `url(${puzzleImg})`}">
        <!-- 右下角缺块: 与游戏空位一致, 大小随棋盘尺寸变化 -->
        <div class="puzzle-goal-blank"></div>
        <div class="puzzle-goal-hint">目标图 · 点击返回拼图</div>
      </div>

      <template v-else>
        <!-- 背景格子 -->
        <div class="puzzle-grid">
          <div v-for="i in game.n * game.n" :key="i" class="puzzle-grid-cell">
            <div class="puzzle-grid-face"></div>
          </div>
        </div>
        <!-- 方块层: transform 平滑滑动, 快速连续操作动画互不干扰 -->
        <div class="puzzle-tiles">
          <div
            v-for="t in game.tiles"
            :key="t.id"
            class="puzzle-tile"
            :style="{'--c': t.c, '--r': t.r}"
            @click.stop="onTileClick(t)"
          >
            <div class="puzzle-tile-face" :style="tileFaceStyle(t.value)"></div>
          </div>
        </div>
      </template>
    </div>

    <div class="puzzle-status">{{ statusText }}</div>

    <div class="puzzle-controls">
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
      <a-button :type="game.view === 'goal' ? 'primary' : 'default'" @click="game.toggleView()">
        <template #icon>
          <eye-outlined />
        </template>
        {{ game.view === 'goal' ? '返回拼图' : '目标图' }}
      </a-button>
      <div class="puzzle-ai-toggle" :class="{'ai-on': game.aiEnabled}">
        <thunderbolt-outlined />
        <span class="puzzle-ai-label">AI 托管</span>
        <a-switch
          size="small"
          :checked="game.aiEnabled"
          checked-children="开"
          un-checked-children="关"
          @change="game.setAiEnabled"
        />
      </div>
    </div>

    <div class="puzzle-size-row">
      <span class="puzzle-size-label">棋盘尺寸</span>
      <a-radio-group size="small" :value="game.n" @change="onSizeChange">
        <a-radio-button :value="3">3x3</a-radio-button>
        <a-radio-button :value="4">4x4</a-radio-button>
      </a-radio-group>
    </div>
  </div>
</template>

<style scoped>
.puzzle {
  display: flex;
  flex-direction: column;
  gap: 12px;
  width: min(100%, 420px);
  margin: 0 auto;
  user-select: none;
}

.puzzle-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  flex-wrap: wrap; /* 窄屏时分值盒换行, 不挤压标题 */
}

.puzzle-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 18px;
  font-weight: 700;
  white-space: nowrap; /* 标题不折行 */
}

.puzzle-badge {
  font-size: 11px;
  font-weight: 500;
  padding: 2px 8px;
  border-radius: 10px;
  color: #1677ff;
  background: rgba(22, 119, 255, 0.12);
  white-space: nowrap; /* 徽章不折行 */
  flex-shrink: 0;
}

.puzzle-scores {
  display: flex;
  gap: 8px;
}

.puzzle-score-box {
  min-width: 52px;
  padding: 4px 10px;
  border-radius: 8px;
  text-align: center;
  background: #3a3b45;
  color: #f7f4f0;
  display: flex;
  flex-direction: column;
  line-height: 1.3;
}

.puzzle-score-box span {
  font-size: 11px;
  opacity: 0.85;
}

.puzzle-score-box strong {
  font-size: 16px;
}

.puzzle-board {
  position: relative;
  aspect-ratio: 1;
  border-radius: 12px;
  background: #33343d;
  touch-action: none; /* 移动端滑动时避免页面滚动 */
  overflow: hidden;
}

.puzzle-board.is-goal {
  cursor: pointer;
}

.puzzle-board.is-solved {
  outline: 2px solid #1677ff;
  outline-offset: -2px;
}

.puzzle-grid {
  position: absolute;
  inset: 0;
  display: grid;
  grid-template-columns: repeat(var(--n), 1fr);
}

.puzzle-grid-cell {
  aspect-ratio: 1;
  padding: 3px;
  box-sizing: border-box;
}

.puzzle-grid-face {
  width: 100%;
  height: 100%;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.08);
}

.puzzle-tiles {
  position: absolute;
  inset: 0;
}

.puzzle-tile {
  position: absolute;
  width: calc(100% / var(--n));
  height: calc(100% / var(--n));
  box-sizing: border-box;
  padding: 3px;
  cursor: pointer;
  transition: transform 0.12s ease;
  transform: translate(calc(var(--c) * 100%), calc(var(--r) * 100%));
}

.puzzle-tile-face {
  width: 100%;
  height: 100%;
  border-radius: 8px;
  background-repeat: no-repeat;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.35);
}

.puzzle-goal {
  position: absolute;
  inset: 0;
  background-size: 100% 100%;
  background-position: center;
}

.puzzle-goal-blank {
  position: absolute;
  right: 3px;
  bottom: 3px;
  width: calc(100% / var(--n) - 6px);
  height: calc(100% / var(--n) - 6px);
  border-radius: 8px;
  background: #33343d;
}

.puzzle-goal-hint {
  position: absolute;
  left: 50%;
  top: 10px;
  transform: translateX(-50%);
  padding: 3px 12px;
  border-radius: 12px;
  font-size: 12px;
  color: #fff;
  background: rgba(0, 0, 0, 0.55);
  white-space: nowrap;
}

.puzzle-status {
  min-height: 20px;
  font-size: 13px;
  text-align: center;
  opacity: 0.75;
  text-wrap: balance; /* 窄屏换行更均匀, 不留孤字 */
}

.puzzle-controls {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  flex-wrap: wrap;
}

.puzzle-ai-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.08);
  transition: background 0.2s;
}

.puzzle-ai-toggle.ai-on {
  background: rgba(22, 119, 255, 0.16);
  color: #1677ff;
}

.puzzle-ai-label {
  font-size: 13px;
}

.puzzle-size-row {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.puzzle-size-label {
  font-size: 12px;
  opacity: 0.75;
}

@media (max-width: 480px) {
  .puzzle-title {
    font-size: 16px;
  }

  .puzzle-score-box {
    min-width: 44px;
    padding: 3px 8px;
  }

  .puzzle-score-box strong {
    font-size: 15px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .puzzle-tile {
    transition: none;
  }
}
</style>
