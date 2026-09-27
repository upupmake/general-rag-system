import {computed, ref} from 'vue'
import {defineStore} from 'pinia'
import {BLANK, isSolved, shuffled} from '@/utils/puzzleSolver'

const AI_INTERVAL = 260 // AI 托管每步动画间隔(ms)
const MAX_HISTORY = 10 // 撤销仅保留最近 10 步

function tilesFromState(state, n) {
  const tiles = []
  for (let i = 0; i < state.length; i++) {
    const v = state[i]
    if (v === BLANK) continue
    tiles.push({id: v, value: v, r: Math.floor(i / n), c: i % n})
  }
  return tiles
}

export const useGamePuzzleStore = defineStore('gamePuzzle', () => {
  const n = ref(4) // 棋盘尺寸, 可选 3x3 / 4x4
  const tiles = ref([]) // [{id, value, r, c}], 空位无 tile
  const moves = ref(0)
  const history = ref([]) // [{tiles, moves}], 仅保留最近 MAX_HISTORY 步
  const view = ref('game') // 'game' 拼图中 | 'goal' 查看目标图
  const aiEnabled = ref(false) // AI 托管开关
  const active = ref(false) // 游戏弹窗是否打开(可见)
  const thinking = ref(false) // AI 是否在求解
  let aiPlan = null // 待执行的移动序列(方块编号)
  let aiTimer = null
  let worker = null
  let solveId = 0
  const savedGames = {} // 各尺寸的游戏快照: 切换 3x3/4x4 时保留各自状态

  const state = computed(() => {
    const s = Array(n.value * n.value).fill(BLANK)
    for (const t of tiles.value) s[t.r * n.value + t.c] = t.value
    return s
  })
  const solved = computed(() => tiles.value.length > 0 && isSolved(state.value))

  function newGame() {
    aiEnabled.value = false // 新游戏重置 AI 托管开关
    stopSolve()
    resetBoard()
  }

  function resetBoard() {
    tiles.value = tilesFromState(shuffled(n.value), n.value)
    moves.value = 0
    history.value = []
    view.value = 'game'
    clearPlan()
    scheduleAi()
  }

  // 切换尺寸保留各自的游戏状态(含撤销历史); 只有"新游戏"才重置当前尺寸
  function setSize(size) {
    const target = Number(size)
    if (target === n.value) return
    aiEnabled.value = false // 切换棋盘同样重置 AI 托管开关
    stopSolve()
    savedGames[n.value] = {
      tiles: tiles.value.map(t => ({...t})),
      moves: moves.value,
      history: history.value,
      view: view.value,
    }
    n.value = target
    const snap = savedGames[target]
    if (!snap) {
      resetBoard()
      return
    }
    tiles.value = snap.tiles.map(t => ({...t}))
    moves.value = snap.moves
    history.value = snap.history
    view.value = snap.view
    clearPlan()
    scheduleAi()
  }

  // 点击空白格旁的方块: 自动与空白交换(规则要求相邻, 其余点击无效)
  function moveTile(value, fromAi = false) {
    if (solved.value) return false
    const size = n.value
    const st = state.value
    const ti = st.indexOf(value)
    const bi = st.indexOf(BLANK)
    const tr = Math.floor(ti / size)
    const tc = ti % size
    const br = Math.floor(bi / size)
    const bc = bi % size
    if (Math.abs(tr - br) + Math.abs(tc - bc) !== 1) return false
    history.value = [...history.value.slice(-(MAX_HISTORY - 1)), {
      tiles: tiles.value.map(t => ({...t})),
      moves: moves.value,
    }]
    for (const t of tiles.value) {
      if (t.value === value) {
        t.r = br
        t.c = bc
      }
    }
    moves.value++
    if (!fromAi) clearPlan() // 玩家手动操作使 AI 计划作废
    scheduleAi()
    return true
  }

  // 空位向 (dr,dc) 滑动一格(方向键/滑动手势)
  function moveBlank(dr, dc) {
    const size = n.value
    const st = state.value
    const bi = st.indexOf(BLANK)
    const r = Math.floor(bi / size) + dr
    const c = (bi % size) + dc
    if (r < 0 || r >= size || c < 0 || c >= size) return false
    return moveTile(st[r * size + c])
  }

  function undo() {
    const prev = history.value.pop()
    if (!prev) return false
    tiles.value = prev.tiles.map(t => ({...t}))
    moves.value = prev.moves
    clearPlan()
    scheduleAi()
    return true
  }

  // 点击切换 当前拼图状态 / 目标完整图
  function toggleView() {
    view.value = view.value === 'game' ? 'goal' : 'game'
    scheduleAi()
  }

  // ---------- AI 托管 ----------
  function setAiEnabled(enabled) {
    aiEnabled.value = enabled
    if (!enabled) clearPlan()
    scheduleAi()
  }

  // 弹窗可见性: 打开时若开关仍开则继续托管; 关闭时强制关停 AI(含 Worker), 游戏状态保留
  function setActive(v) {
    active.value = v
    if (!v) {
      aiEnabled.value = false
      stopSolve()
      clearPlan()
    }
    scheduleAi()
  }

  function clearPlan() {
    solveId++ // 使在途求解结果过期
    aiPlan = null
    thinking.value = false
  }

  function stopSolve() {
    solveId++
    thinking.value = false
    if (worker) {
      worker.terminate()
      worker = null
    }
  }

  function scheduleAi() {
    if (aiTimer) {
      clearTimeout(aiTimer)
      aiTimer = null
    }
    if (aiEnabled.value && active.value && !solved.value && view.value === 'game') {
      aiTimer = setTimeout(aiStep, AI_INTERVAL)
    }
  }

  function aiStep() {
    if (!aiEnabled.value || !active.value || solved.value || view.value !== 'game') return
    if (aiPlan && aiPlan.length) {
      moveTile(aiPlan.shift(), true)
      return
    }
    requestSolution()
  }

  function requestSolution() {
    if (thinking.value) return
    if (!worker) {
      worker = new Worker(
        new URL('../workers/puzzleAi.worker.js', import.meta.url),
        {type: 'module'}
      )
      worker.onmessage = (e) => {
        const {id, moves: plan} = e.data
        if (id !== solveId) return // 过期结果, 丢弃
        thinking.value = false
        aiPlan = plan && plan.length ? plan : null
        scheduleAi()
      }
    }
    solveId++
    thinking.value = true
    worker.postMessage({id: solveId, state: state.value, n: n.value})
  }

  newGame() // 初始随机打乱

  return {
    n,
    tiles,
    moves,
    history,
    view,
    aiEnabled,
    active,
    thinking,
    solved,
    state,
    newGame,
    setSize,
    moveTile,
    moveBlank,
    undo,
    toggleView,
    setAiEnabled,
    setActive,
  }
})
