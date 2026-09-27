import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import {
  N2,
  EMPTY,
  BLACK,
  WHITE,
  checkWin,
} from '@/utils/gomokuEngine.js'

// 每步思考时间（秒）：对手恒为最强 AI（搜索在 Web Worker 中进行，不阻塞 UI）
const AI_TIME_LIMIT = 1.5
const AI_MAX_DEPTH = 14

export const useGameGomokuStore = defineStore('gameGomoku', () => {
  /** 落子历史 [{cell, side}] */
  const moves = ref([])
  /** 玩家执子（新局随机先后手） */
  const humanSide = ref(BLACK)
  /** 玩家方是否托管给最强 AI */
  const aiEnabled = ref(false)
  const thinking = ref(false)
  /** 0 未分胜负 / BLACK / WHITE / 3 和棋 */
  const winner = ref(0)
  const active = ref(false)
  /** 最近一次 AI 搜索信息 {depth, score, reason, time} */
  const lastInfo = ref(null)

  let worker = null
  let searchId = 0

  const board = computed(() => {
    const b = new Array(N2).fill(EMPTY)
    for (const m of moves.value) b[m.cell] = m.side
    return b
  })
  const currentSide = computed(() =>
    moves.value.length % 2 === 0 ? BLACK : WHITE
  )
  const opponentSide = computed(() => (humanSide.value === BLACK ? WHITE : BLACK))
  const lastMove = computed(() =>
    moves.value.length ? moves.value[moves.value.length - 1].cell : -1
  )
  const gameOver = computed(() => winner.value !== 0)
  const canHumanPlay = computed(
    () =>
      active.value &&
      !gameOver.value &&
      !aiEnabled.value &&
      currentSide.value === humanSide.value
  )
  const isAiTurn = computed(() => {
    const side = currentSide.value
    return (
      !gameOver.value &&
      (side === opponentSide.value || (aiEnabled.value && side === humanSide.value))
    )
  })

  function newGame() {
    stopAi()
    moves.value = []
    winner.value = 0
    lastInfo.value = null
    humanSide.value = Math.random() < 0.5 ? BLACK : WHITE
    maybeAiMove()
  }

  function play(cell) {
    if (gameOver.value || cell < 0 || cell >= N2) return
    const side = currentSide.value
    if (board.value[cell] !== EMPTY) return
    moves.value = [...moves.value, { cell, side }]
    if (checkWin(board.value, cell, side)) {
      winner.value = side
      stopAi()
      return
    }
    if (moves.value.length >= N2) {
      winner.value = 3
      return
    }
    maybeAiMove()
  }

  function maybeAiMove() {
    if (!active.value || !isAiTurn.value) return
    requestAiMove()
  }

  function requestAiMove() {
    if (!worker) {
      worker = new Worker(
        new URL('../workers/gomokuAi.worker.js', import.meta.url),
        { type: 'module' }
      )
      worker.onmessage = (e) => {
        const data = e.data
        if (data.id !== searchId) return // 过期结果，丢弃
        thinking.value = false
        if (data.error) {
          console.error('[gomoku ai]', data.error)
          return
        }
        lastInfo.value = {
          depth: data.depth,
          score: data.score,
          reason: data.reason,
          time: data.time,
        }
        if (data.move != null) play(data.move)
      }
    }
    searchId += 1
    thinking.value = true
    worker.postMessage({
      id: searchId,
      moves: moves.value.map((m) => [m.cell, m.side]),
      side: currentSide.value,
      timeLimit: AI_TIME_LIMIT,
      maxDepth: AI_MAX_DEPTH,
    })
  }

  /** 终止后台计算（关闭弹窗 / 结束棋局 / 新局时调用），保留棋局状态 */
  function stopAi() {
    searchId += 1
    thinking.value = false
    if (worker) {
      worker.terminate()
      worker = null
    }
  }

  function setActive(v) {
    active.value = v
    if (!v) {
      stopAi()
    } else if (!gameOver.value) {
      maybeAiMove()
    }
  }

  function setAiEnabled(v) {
    aiEnabled.value = v
    if (v) {
      maybeAiMove()
    } else if (currentSide.value === humanSide.value) {
      stopAi() // 玩家收回托管，取消为其思考的计算
    }
  }

  return {
    moves,
    humanSide,
    aiEnabled,
    thinking,
    winner,
    active,
    lastInfo,
    board,
    currentSide,
    opponentSide,
    lastMove,
    gameOver,
    canHumanPlay,
    isAiTurn,
    newGame,
    play,
    setActive,
    setAiEnabled,
    stopAi,
  }
})
