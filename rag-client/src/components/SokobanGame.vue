<script setup>
import {computed, onMounted, onUnmounted} from 'vue'
import {ReloadOutlined, UndoOutlined, ThunderboltOutlined} from '@ant-design/icons-vue'
import {useGameSokobanStore} from '@/stores/gameSokoban'
import {getLevels} from '@/utils/sokobanSolver'
// 美术素材: Kenney.nl "Sokoban" 素材包 (CC0 公共领域, 可商用免署名),
// 许可见 src/assets/sokoban/LICENSE-kenney.txt
import wallImg from '@/assets/sokoban/wall.png'
import floorImg from '@/assets/sokoban/floor.png'
import goalImg from '@/assets/sokoban/goal.png'
import crateImg from '@/assets/sokoban/crate.png'
import playerDownImg from '@/assets/sokoban/player_down.png'
import playerUpImg from '@/assets/sokoban/player_up.png'
import playerSideImg from '@/assets/sokoban/player_side.png'

const game = useGameSokobanStore()

const levelOptions = getLevels().map(m => ({
  value: m.index,
  label: `第 ${m.index} 关 · ${m.title}（最优推动 ${m.optimalPushes}）`,
}))

const wallSet = computed(() => new Set(game.level.walls.map(([x, y]) => x + ',' + y)))
const goalSet = computed(() => new Set(game.level.goals.map(([x, y]) => x + ',' + y)))
const floorSet = computed(() => new Set(game.level.floors.map(([x, y]) => x + ',' + y)))

// 静态格子层: 墙/地板/目标（棋盘尺寸不规则, 虚空按地板渲染但不可达, 与墙视觉区分）
const cells = computed(() => {
  const out = []
  for (let y = 0; y < game.level.h; y++) {
    for (let x = 0; x < game.level.w; x++) {
      const k = x + ',' + y
      out.push({
        key: k,
        wall: wallSet.value.has(k),
        goal: goalSet.value.has(k),
        floor: floorSet.value.has(k),
      })
    }
  }
  return out
})

// 玩家贴图按朝向切换: 上=背影, 左右侧共用侧身图(左侧水平镜像)
const playerImg = computed(() => {
  if (game.facing === 'U') return playerUpImg
  if (game.facing === 'L' || game.facing === 'R') return playerSideImg
  return playerDownImg
})

const boardStyle = computed(() => ({
  '--w': game.level.w,
  '--h': game.level.h,
  '--img-wall': `url(${wallImg})`,
  '--img-floor': `url(${floorImg})`,
  '--img-goal': `url(${goalImg})`,
  '--img-crate': `url(${crateImg})`,
  '--img-player': `url(${playerImg.value})`,
}))

function pieceStyle(x, y, flip = false) {
  return {transform: `translate(${x * 100}%, ${y * 100}%)${flip ? ' scaleX(-1)' : ''}`}
}

const statusText = computed(() => {
  if (game.solved) {
    return `已通关! 步数 ${game.moves} · 推动 ${game.pushes}` + (game.winByAi ? '（AI 托管完成）' : '')
  }
  if (game.aiNotice) return game.aiNotice // 无解/超时警示优先, 不依赖托管开关
  if (game.aiInfo && game.aiEnabled) return game.aiInfo
  if (game.thinking) return 'AI 思考中…'
  if (game.aiEnabled) return 'AI 托管中…'
  return '方向键 / WASD 移动, 触屏滑动或方向盘亦可'
})

function onKeydown(e) {
  const t = e.target
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' ||
    (t.closest && t.closest('.ant-select, .ant-select-dropdown')))) return
  const dirs = {
    ArrowUp: 'U', ArrowDown: 'D', ArrowLeft: 'L', ArrowRight: 'R',
    w: 'U', W: 'U', s: 'D', S: 'D', a: 'L', A: 'L', d: 'R', D: 'R',
  }
  const dir = dirs[e.key]
  if (dir) {
    e.preventDefault()
    game.move(dir)
    return
  }
  if (e.key === 'u' || e.key === 'U') game.undo()
  else if (e.key === 'r' || e.key === 'R') game.restart()
}

let touchStart = null

function onTouchStart(e) {
  // 存普通对象: Touch.x/y 别名并不可靠, 必须取 clientX/clientY
  const t = e.touches[0]
  touchStart = {x: t.clientX, y: t.clientY}
}

function onTouchEnd(e) {
  if (!touchStart) {
    touchStart = null
    return
  }
  const dx = e.changedTouches[0].clientX - touchStart.x
  const dy = e.changedTouches[0].clientY - touchStart.y
  touchStart = null
  if (Math.max(Math.abs(dx), Math.abs(dy)) < 32) return // 视为点击, 交给按钮处理
  e.preventDefault() // 滑动后抑制浏览器合成 click
  // 滑动方向 = 玩家移动方向(跟手)
  if (Math.abs(dx) > Math.abs(dy)) game.move(dx > 0 ? 'R' : 'L')
  else game.move(dy > 0 ? 'D' : 'U')
}

onMounted(() => window.addEventListener('keydown', onKeydown))
onUnmounted(() => window.removeEventListener('keydown', onKeydown))
</script>

<template>
  <div class="sokoban">
    <div class="sokoban-header">
      <div class="sokoban-title">
        推箱子
      </div>
      <div class="sokoban-scores">
        <div class="sokoban-score-box">
          <span>步数</span>
          <strong>{{ game.moves }}</strong>
        </div>
        <div class="sokoban-score-box">
          <span>推动</span>
          <strong>{{ game.pushes }}</strong>
        </div>
        <div class="sokoban-score-box">
          <span>最优推动</span>
          <strong>{{ game.level.optimalPushes ?? '-' }}</strong>
        </div>
      </div>
    </div>

    <!-- 棋盘: 静态格子层 + 箱子/玩家移动层(transform 平滑滑动, 快速连续操作互不干扰) -->
    <div
      class="sokoban-board"
      :class="{'is-solved': game.solved}"
      :style="boardStyle"
      @touchstart="onTouchStart"
      @touchend="onTouchEnd"
    >
      <div class="sokoban-grid">
        <div
          v-for="c in cells"
          :key="c.key"
          class="sokoban-cell"
          :class="{'is-wall': c.wall, 'is-goal': c.goal, 'is-floor': c.floor}"
        ></div>
      </div>
      <div
        v-for="b in game.boxes"
        :key="'b' + b.id"
        class="sokoban-box"
        :class="{'is-done': goalSet.has(b.x + ',' + b.y)}"
        :style="pieceStyle(b.x, b.y)"
      ></div>
      <div
        class="sokoban-player"
        :class="{'is-done': goalSet.has(game.player[0] + ',' + game.player[1])}"
        :style="pieceStyle(game.player[0], game.player[1], game.facing === 'L')"
      ></div>
    </div>

    <div class="sokoban-status" :class="{'is-warning': game.aiNotice}">{{ statusText }}</div>

    <div class="sokoban-controls">
      <a-button type="primary" @click="game.restart()">
        <template #icon>
          <reload-outlined />
        </template>
        重开
      </a-button>
      <a-button :disabled="!game.history.length" @click="game.undo()">
        <template #icon>
          <undo-outlined />
        </template>
        撤销
      </a-button>
      <a-button @click="game.prevLevel()">上一关</a-button>
      <a-button @click="game.nextLevel()">下一关</a-button>
      <div class="sokoban-ai-toggle" :class="{'ai-on': game.aiEnabled}">
        <thunderbolt-outlined />
        <span class="sokoban-ai-label">AI 托管</span>
        <a-switch
          size="small"
          :checked="game.aiEnabled"
          checked-children="开"
          un-checked-children="关"
          @change="game.setAiEnabled"
        />
      </div>
    </div>

    <div class="sokoban-level-row">
      <span class="sokoban-level-label">关卡</span>
      <a-select
        size="small"
        class="sokoban-level-select"
        :value="game.levelIndex"
        :options="levelOptions"
        @change="game.setLevel"
      />
    </div>

    <!-- 方向盘: 移动端/点击操作 -->
    <div class="sokoban-dpad">
      <button class="sokoban-dpad-btn dpad-up" @click.prevent="game.move('U')">↑</button>
      <button class="sokoban-dpad-btn dpad-left" @click.prevent="game.move('L')">←</button>
      <button class="sokoban-dpad-btn dpad-down" @click.prevent="game.move('D')">↓</button>
      <button class="sokoban-dpad-btn dpad-right" @click.prevent="game.move('R')">→</button>
    </div>

    <div class="sokoban-credit">美术素材: Kenney.nl · Sokoban Pack（CC0）</div>
  </div>
</template>

<style scoped>
.sokoban {
  display: flex;
  flex-direction: column;
  gap: 12px;
  width: min(100%, 560px);
  max-height: calc(100vh - 104px); /* 矮窗口兜底: 内部滚动, 弹窗永不被视口裁切(预留 48px 弹窗内边距+40px 内容边距+16px 余量) */
  max-height: calc(100dvh - 104px);
  overflow-y: auto;
  margin: 0 auto;
  user-select: none;
}

.sokoban-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  flex-wrap: wrap; /* 窄屏时分值盒换行, 不挤压标题 */
}

.sokoban-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 18px;
  font-weight: 700;
  white-space: nowrap;
}

.sokoban-scores {
  display: flex;
  gap: 8px;
}

.sokoban-score-box {
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

.sokoban-score-box span {
  font-size: 11px;
  opacity: 0.85;
  white-space: nowrap;
}

.sokoban-score-box strong {
  font-size: 16px;
}

.sokoban-board {
  position: relative;
  /* 宽度受高度预算约束(视口高度-界面件高度): 棋盘+控件整体一屏放下, 手机上方向盘不再被顶出视口 */
  width: min(100%, 520px, calc(max(180px, 100vh - 420px) * var(--w) / var(--h)));
  width: min(100%, 520px, calc(max(180px, 100dvh - 420px) * var(--w) / var(--h)));
  margin: 0 auto;
  aspect-ratio: var(--w) / var(--h); /* 棋盘不规则(宽高不同), 按关卡尺寸定比例 */
  flex-shrink: 0; /* 禁止被 flex 压扁: 压扁会破坏宽高比导致贴图变形, 超高交给容器滚动 */
  border-radius: 12px;
  background: #e6eaf0; /* 虚空区(关卡轮廓外): 浅色中性, 与明亮主题协调 */
  border: 1px solid #d5dae2;
  touch-action: none; /* 移动端滑动时避免页面滚动 */
  overflow: hidden;
}

.sokoban-board.is-solved {
  outline: 2px solid #1677ff;
  outline-offset: -2px;
}

.sokoban-grid {
  position: absolute;
  inset: 0;
  display: grid;
  /* 显式声明行/列两组轨道: 隐式 auto 行会被内容撑高 */
  grid-template-columns: repeat(var(--w), minmax(0, 1fr));
  grid-template-rows: repeat(var(--h), minmax(0, 1fr));
}

.sokoban-cell {
  box-sizing: border-box;
  background: transparent;
}

.sokoban-cell.is-wall {
  background: var(--img-wall) center / 100% 100% no-repeat;
  /* 深红砖: 压暗提饱和, 与浅琥珀木箱拉开明度与色相差 */
  filter: saturate(1.3) brightness(0.9);
}

/* 注意层叠顺序: 目标格同时带 is-floor, 必须让 is-goal 在后覆盖地板图 */
.sokoban-cell.is-floor {
  background: var(--img-floor) center / 100% 100% no-repeat;
  /* 素材原色偏青绿(RGB 123,147,150), 灰度化保证零色相(纯中性灰), 避免任何混色 */
  filter: grayscale(1) brightness(1.12);
}

.sokoban-cell.is-goal {
  background: var(--img-goal) center / 100% 100% no-repeat;
}

.sokoban-box,
.sokoban-player {
  position: absolute;
  top: 0;
  left: 0;
  width: calc(100% / var(--w));
  height: calc(100% / var(--h));
  box-sizing: border-box;
  transition: transform 0.12s ease;
}

/* 注意: 百分比 padding 相对父级(棋盘)宽度解析会把棋子挤扁, 必须用绝对定位 inset(相对格子) */
.sokoban-box::before {
  content: '';
  position: absolute;
  inset: 5%;
  background: var(--img-crate) center / 100% 100% no-repeat;
  /* 浅琥珀木箱: 比深红墙亮一档, 两者不再同为暖棕难分 */
  filter: drop-shadow(0 2px 3px rgba(0, 0, 0, 0.4)) brightness(1.16) saturate(1.05);
}

.sokoban-box.is-done::before {
  border-radius: 12%;
  box-shadow: 0 0 0 2px #2fbf71, 0 0 8px rgba(47, 191, 113, 0.6);
  /* 同一素材滤镜变绿: 与普通箱绝对同形同大, 色相+光圈表状态 */
  filter: drop-shadow(0 2px 3px rgba(0, 0, 0, 0.4)) hue-rotate(105deg) saturate(1.4) brightness(1.08);
}

.sokoban-player::before {
  content: '';
  position: absolute;
  inset: 3%;
  background: var(--img-player) center / contain no-repeat;
  filter: drop-shadow(0 2px 3px rgba(0, 0, 0, 0.45));
}

.sokoban-player.is-done::before {
  filter: drop-shadow(0 2px 3px rgba(0, 0, 0, 0.45)) drop-shadow(0 0 6px rgba(242, 177, 52, 0.9));
}

.sokoban-status {
  min-height: 20px;
  font-size: 13px;
  text-align: center;
  opacity: 0.75;
  text-wrap: balance; /* 窄屏换行更均匀, 不留孤字 */
}

.sokoban-status.is-warning {
  color: #d46b08;
  opacity: 1;
  font-weight: 600;
}

.sokoban-controls {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  flex-wrap: wrap;
}

.sokoban-ai-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 8px;
  border: 1px solid #d9d9d9;
  background: #f5f5f5;
  transition: background 0.2s, border-color 0.2s;
}

.sokoban-ai-toggle.ai-on {
  background: rgba(22, 119, 255, 0.1);
  border-color: rgba(22, 119, 255, 0.45);
  color: #1677ff;
}

.sokoban-ai-label {
  font-size: 13px;
}

.sokoban-level-row {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.sokoban-level-label {
  font-size: 12px;
  opacity: 0.75;
}

.sokoban-level-select {
  min-width: 240px;
  max-width: 100%;
}

.sokoban-dpad {
  display: grid;
  grid-template-columns: repeat(3, 56px);
  grid-template-rows: repeat(2, 44px);
  gap: 6px;
  justify-content: center;
}

.sokoban-dpad-btn {
  border: 1px solid #d9d9d9;
  background: #f5f5f5;
  color: rgba(0, 0, 0, 0.85);
  border-radius: 8px;
  font-size: 16px;
  line-height: 1;
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s;
}

.sokoban-dpad-btn:hover {
  background: rgba(22, 119, 255, 0.1);
  border-color: rgba(22, 119, 255, 0.45);
}

.sokoban-dpad-btn:active {
  background: rgba(22, 119, 255, 0.22);
}

.dpad-up {
  grid-column: 2;
  grid-row: 1;
}

.dpad-left {
  grid-column: 1;
  grid-row: 2;
}

.dpad-down {
  grid-column: 2;
  grid-row: 2;
}

.dpad-right {
  grid-column: 3;
  grid-row: 2;
}

.sokoban-credit {
  font-size: 11px;
  opacity: 0.45;
  text-align: center;
}

@media (max-width: 768px) {
  .sokoban {
    gap: 8px;
  }

  .sokoban-title {
    font-size: 16px;
  }

  .sokoban-score-box {
    min-width: 44px;
    padding: 3px 8px;
  }

  .sokoban-score-box strong {
    font-size: 15px;
  }

  .sokoban-level-select {
    min-width: 180px;
  }

  .sokoban-dpad {
    grid-template-columns: repeat(3, 52px);
    grid-template-rows: repeat(2, 42px);
    gap: 5px;
  }

  .sokoban-status {
    min-height: 16px;
    font-size: 12px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .sokoban-box,
  .sokoban-player {
    transition: none;
  }
}
</style>
