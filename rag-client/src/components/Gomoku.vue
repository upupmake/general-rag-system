<script setup>
import {computed, onMounted, ref, watch} from 'vue'
import {useGameGomokuStore} from '@/stores/gameGomoku'
import {ReloadOutlined, ThunderboltOutlined} from '@ant-design/icons-vue'

const game = useGameGomokuStore()

onMounted(() => {
  if (game.moves.length === 0) game.newGame()
})

// 触屏两段式落子: 第一次轻触选点(大标记), 再次点击同点确认, 防小屏误触
const selectedCell = ref(-1)
watch(() => [game.moves.length, game.gameOver], () => {
  selectedCell.value = -1
})

const sideText = computed(() =>
  game.humanSide === 1 ? '你执黑 · 先手' : '你执白 · 后手'
)

const statusText = computed(() => {
  if (game.gameOver) {
    if (game.winner === 3) return '和棋! 点「新游戏」再来一局'
    return `${game.winner === game.humanSide ? '你赢了!' : 'AI 获胜!'} 点「新游戏」再来一局`
  }
  if (selectedCell.value >= 0) return '再次点击同一位置确认落子'
  if (game.thinking) return 'AI 思考中…'
  if (game.aiEnabled) return 'AI 托管中, 正在替你落子…'
  return '点击棋盘落子 (PC / 手机通用)'
})

function onCellClick(idx) {
  if (!game.canHumanPlay || game.board[idx] !== 0) return
  game.play(idx)
}

function onCellTouchEnd(idx, e) {
  e.preventDefault() // 阻止合成 click, 触屏走两段式确认
  if (!game.canHumanPlay || game.board[idx] !== 0) return
  if (selectedCell.value === idx) {
    selectedCell.value = -1
    game.play(idx)
  } else {
    selectedCell.value = idx
  }
}
</script>

<template>
  <div class="gomoku">
    <div class="gomoku-header">
      <div class="gomoku-title">
        五子棋
      </div>
      <div class="gomoku-side" :class="{ 'side-black': game.humanSide === 1 }">
        {{ sideText }}
      </div>
    </div>

    <div class="gomoku-board">
      <div
        v-for="(v, idx) in game.board"
        :key="idx"
        class="gomoku-cell"
        :class="{ 'can-play': game.canHumanPlay && v === 0 }"
        @click="onCellClick(idx)"
        @touchend="onCellTouchEnd(idx, $event)"
      >
        <span
          v-if="v"
          class="gomoku-stone"
          :class="{
            'stone-black': v === 1,
            'stone-white': v === 2,
            'is-last': idx === game.lastMove,
          }"
        ></span>
        <span
          v-else-if="idx === selectedCell"
          class="gomoku-stone gomoku-selected"
          :class="game.currentSide === 1 ? 'stone-black' : 'stone-white'"
        ></span>
        <span
          v-else-if="game.canHumanPlay"
          class="gomoku-stone gomoku-preview"
          :class="game.currentSide === 1 ? 'stone-black' : 'stone-white'"
        ></span>
      </div>
    </div>

    <div class="gomoku-status">{{ statusText }}</div>

    <div class="gomoku-controls">
      <a-button type="primary" @click="game.newGame()">
        <template #icon>
          <reload-outlined />
        </template>
        新游戏
      </a-button>
      <div class="gomoku-ai-toggle" :class="{ 'ai-on': game.aiEnabled }">
        <thunderbolt-outlined />
        <span class="gomoku-ai-label">AI 托管(我方)</span>
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
.gomoku {
  display: flex;
  flex-direction: column;
  gap: 12px;
  width: 100%;
  margin: 0 auto;
  user-select: none;
}

.gomoku-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  padding-right: 40px; /* 给弹窗右上角关闭按钮留位 */
}

.gomoku-title {
  font-size: 22px;
  font-weight: 700;
  color: #776e65;
  display: flex;
  align-items: center;
  gap: 8px;
}

[data-theme='dark'] .gomoku-title {
  color: #e8e6e3;
}

.gomoku-side {
  font-size: 12px;
  padding: 3px 10px;
  border-radius: 10px;
  color: #8c8c8c;
  border: 1px solid rgba(0, 0, 0, 0.12);
}

[data-theme='dark'] .gomoku-side {
  color: #999;
  border-color: rgba(255, 255, 255, 0.14);
}

.gomoku-side.side-black {
  color: #d48a3c;
  border-color: rgba(212, 138, 60, 0.5);
}

.gomoku-board {
  display: grid;
  /* 行列都显式等分: 若行留 auto, 有子的行会被棋子的最小内容高度撑高, 导致网格不齐 */
  grid-template-columns: repeat(15, minmax(0, 1fr));
  grid-template-rows: repeat(15, minmax(0, 1fr));
  aspect-ratio: 1;
  border-radius: 12px;
  border-right: 1px solid #a97c2d;
  border-bottom: 1px solid #a97c2d;
  background: #dcb468;
  overflow: hidden;
  touch-action: manipulation; /* 移动端点击落子, 避免双击缩放延迟 */
}

[data-theme='dark'] .gomoku-board {
  background: #5c4a2e;
  border-right-color: #2f2617;
  border-bottom-color: #2f2617;
}

.gomoku-cell {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  border-top: 1px solid #a97c2d;
  border-left: 1px solid #a97c2d;
}

[data-theme='dark'] .gomoku-cell {
  border-top-color: #2f2617;
  border-left-color: #2f2617;
}

.gomoku-cell.can-play {
  cursor: pointer;
}

.gomoku-stone {
  width: 84%;
  aspect-ratio: 1;
  border-radius: 50%;
  animation: gomoku-pop 0.12s ease-out;
}

.stone-black {
  background: radial-gradient(circle at 34% 28%, #6b6b6b, #101010 70%);
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.35);
}

.stone-white {
  background: radial-gradient(circle at 34% 28%, #ffffff, #c2c2c2 72%);
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.25);
}

.gomoku-stone.is-last::after {
  content: '';
  position: absolute;
  inset: 0;
  margin: auto;
  width: 26%;
  aspect-ratio: 1;
  border-radius: 50%;
  background: #f5222d;
}

.gomoku-preview {
  opacity: 0;
  transition: opacity 0.12s ease;
  animation: none; /* 预览点挂载时不播放入场动画, 否则轮到玩家时满屏闪烁棋子 */
}

.gomoku-selected {
  opacity: 0.65;
  box-shadow: 0 0 0 2px rgba(22, 119, 255, 0.9);
  animation: none;
}

.gomoku-cell.can-play:hover .gomoku-preview {
  opacity: 0.35;
}

@keyframes gomoku-pop {
  from {
    transform: scale(0.8);
    opacity: 0.5;
  }
  to {
    transform: scale(1);
    opacity: 1;
  }
}

@media (prefers-reduced-motion: reduce) {
  .gomoku-stone {
    animation: none;
  }

  .gomoku-preview {
    transition: none;
  }
}

.gomoku-status {
  min-height: 20px;
  font-size: 13px;
  color: #8c8c8c;
  text-align: center;
}

[data-theme='dark'] .gomoku-status {
  color: #999;
}

.gomoku-controls {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.gomoku-ai-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-left: auto;
  padding: 4px 10px;
  border-radius: 8px;
  border: 1px solid rgba(0, 0, 0, 0.1);
  transition: all 0.2s;
}

[data-theme='dark'] .gomoku-ai-toggle {
  border-color: rgba(255, 255, 255, 0.14);
}

.gomoku-ai-toggle.ai-on {
  color: #faad14;
  border-color: rgba(250, 173, 20, 0.5);
  background: rgba(250, 173, 20, 0.08);
}

.gomoku-ai-label {
  font-size: 13px;
}

/* 移动端: 更紧凑的间距与更大的触控目标 */
@media (max-width: 768px) {
  .gomoku {
    gap: 10px;
  }

  .gomoku-board {
    /* 吃掉弹窗 content/body 的横向内边距, 棋盘满宽最大化格子尺寸 */
    width: calc(100% + 64px);
    margin: 0 -32px;
    border-radius: 0;
  }

  .gomoku-controls :deep(.ant-btn) {
    flex: 1;
    min-height: 38px;
  }

  .gomoku-ai-toggle {
    width: 100%;
    margin-left: 0;
    justify-content: center;
    min-height: 38px;
  }
}
</style>
