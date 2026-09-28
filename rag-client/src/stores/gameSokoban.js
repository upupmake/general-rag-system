import {computed, ref} from 'vue'
import {defineStore} from 'pinia'
import {getLevel, getLevels, isSolved, tryMove} from '@/utils/sokobanSolver'

const AI_INTERVAL = 150 // AI 托管每步动画间隔(ms), 与终端版 0.15s/步一致
const MAX_HISTORY = 10 // 撤销仅保留最近 10 步
const SOLVE_BUDGET_MS = 30000 // AI 求解时间预算(ms), 对应终端版 --budget 30

// 单步最多一个箱子移动: 位置不变的箱子保留 id, 移动的箱子配对到新位置,
// 保证渲染层 transform 动画连贯（key 稳定）
function reassignIds(oldBoxes, newCells) {
  const key = (x, y) => x + ',' + y
  const remaining = new Set(newCells.map(([x, y]) => key(x, y)))
  const kept = []
  const moved = []
  for (const b of oldBoxes) {
    if (remaining.delete(key(b.x, b.y))) kept.push(b)
    else moved.push(b)
  }
  const left = [...remaining]
  for (let i = 0; i < moved.length; i++) {
    const [x, y] = left[i].split(',').map(Number) // left 是 "x,y" 字符串键, 需还原成数字坐标
    kept.push({id: moved[i].id, x, y})
  }
  return kept
}

export const useGameSokobanStore = defineStore('gameSokoban', () => {
  const levelIndex = ref(1)
  const boxes = ref([]) // [{id, x, y}], id 稳定供动画; 求解器接口用 boxCells
  const player = ref([0, 0])
  const facing = ref('D') // 玩家朝向 'U'/'D'/'L'/'R', 供渲染层切换方向贴图
  const moves = ref(0)
  const pushes = ref(0)
  const history = ref([]) // [{boxes, player, moves, pushes}], 仅保留最近 MAX_HISTORY 步
  const aiEnabled = ref(false) // AI 托管开关
  const active = ref(false) // 游戏弹窗是否打开(可见)
  const thinking = ref(false) // AI 是否在求解
  const winByAi = ref(false) // 本关是否由 AI 托管完成
  const aiInfo = ref('') // 最近一次 AI 求解摘要(推动数/最优性/超时提示)
  const aiNotice = ref('') // 无解/超时警示(独立于托管开关显示, 局面变化即清除)
  let aiPlan = null // 待执行的移动序列('U'/'D'/'L'/'R')
  let aiTimer = null
  let worker = null
  let solveId = 0
  let nextBoxId = 1

  const level = computed(() => getLevel(levelIndex.value))
  const boxCells = computed(() => boxes.value.map(b => [b.x, b.y]))
  const solved = computed(() => boxes.value.length > 0 && isSolved(level.value, boxCells.value))

  function snapshot() {
    return {
      boxes: boxes.value.map(b => ({...b})),
      player: [...player.value],
      facing: facing.value,
      moves: moves.value,
      pushes: pushes.value,
    }
  }

  function resetBoard() {
    nextBoxId = 1
    boxes.value = level.value.startBoxes.map(([x, y]) => ({id: nextBoxId++, x, y}))
    player.value = [...level.value.startPlayer]
    facing.value = 'D'
    moves.value = 0
    pushes.value = 0
    history.value = []
    winByAi.value = false
    aiInfo.value = ''
    aiNotice.value = ''
    clearPlan()
    scheduleAi()
  }

  // 重开本关: 重置 AI 托管开关(与"新游戏"语义一致)
  function restart() {
    aiEnabled.value = false
    stopSolve()
    resetBoard()
  }

  // 切关(含上一关/下一关): 重置 AI 托管开关, 重置该关棋盘
  function setLevel(index) {
    const target = Number(index)
    if (!target || target === levelIndex.value) return
    aiEnabled.value = false
    stopSolve()
    levelIndex.value = target
    resetBoard()
  }

  function nextLevel() {
    const n = getLevels().length
    setLevel((levelIndex.value % n) + 1)
  }

  function prevLevel() {
    const n = getLevels().length
    setLevel(((levelIndex.value - 2 + n) % n) + 1)
  }

  // 走/推一步。fromAi=true 表示托管回放(不作废 AI 计划)
  function move(dir, fromAi = false) {
    if (solved.value) return false
    const r = tryMove(level.value, boxCells.value, player.value, dir)
    if (!r) return false
    history.value = [...history.value.slice(-(MAX_HISTORY - 1)), snapshot()]
    boxes.value = reassignIds(boxes.value, r.boxes)
    player.value = r.player
    facing.value = dir
    moves.value++
    if (r.pushed) pushes.value++
    winByAi.value = fromAi // 通关归属以最后一步为准
    if (!fromAi) clearPlan() // 玩家手动操作使 AI 计划作废
    scheduleAi()
    return true
  }

  function undo() {
    const prev = history.value.pop()
    if (!prev) return false
    boxes.value = prev.boxes.map(b => ({...b}))
    player.value = [...prev.player]
    facing.value = prev.facing
    moves.value = prev.moves
    pushes.value = prev.pushes
    winByAi.value = false
    clearPlan()
    scheduleAi()
    return true
  }

  // ---------- AI 托管 ----------
  function setAiEnabled(enabled) {
    aiEnabled.value = enabled
    if (!enabled) clearPlan()
    else aiInfo.value = ''
    scheduleAi()
  }

  // 弹窗可见性: 关闭时强制关停 AI(含 Worker), 游戏状态保留
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
    aiNotice.value = '' // 局面变化(走动/撤销/关闭弹窗)后旧的无解/超时提示不再适用
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
    if (aiEnabled.value && active.value && !solved.value) {
      aiTimer = setTimeout(aiStep, AI_INTERVAL)
    }
  }

  function aiStep() {
    if (!aiEnabled.value || !active.value || solved.value) return
    if (aiPlan && aiPlan.length) {
      move(aiPlan.shift(), true)
      return
    }
    requestSolution()
  }

  function requestSolution() {
    if (thinking.value) return
    if (!worker) {
      worker = new Worker(
        new URL('../workers/sokobanAi.worker.js', import.meta.url),
        {type: 'module'}
      )
      worker.onmessage = (e) => {
        const d = e.data
        if (d.id !== solveId) return // 过期结果, 丢弃
        thinking.value = false
        if (d.moves) {
          aiPlan = [...d.moves]
          aiInfo.value = `AI: 推动 ${d.pushCount} 步${d.optimal ? '(已证最优)' : '(尽力而为)'}`
        } else {
          aiPlan = null
          aiInfo.value = ''
          aiNotice.value = d.solvable
            ? `AI 求解超时, 未找到解 — 可撤销或重开再试`
            : `当前局面无解(箱子死锁) — 可撤销或重开再试`
          aiEnabled.value = false // 无解/超时自动关闭托管, 不卡死界面
        }
        scheduleAi()
      }
    }
    solveId++
    thinking.value = true
    worker.postMessage({
      id: solveId,
      // 必须发纯数据副本: 响应式 Proxy 数组无法 structured clone(会抛 DataCloneError)
      level: getLevel(levelIndex.value),
      boxes: boxCells.value.map(c => [c[0], c[1]]),
      player: [player.value[0], player.value[1]],
      budgetMs: SOLVE_BUDGET_MS,
    })
  }

  resetBoard() // 初始第 1 关

  return {
    levelIndex,
    level,
    boxes,
    player,
    facing,
    moves,
    pushes,
    history,
    aiEnabled,
    active,
    thinking,
    solved,
    winByAi,
    aiInfo,
    aiNotice,
    move,
    undo,
    restart,
    setLevel,
    nextLevel,
    prevLevel,
    setAiEnabled,
    setActive,
  }
})
