import {computed, ref} from 'vue'
import {defineStore} from 'pinia'

const SIZE = 4
const BEST_KEY = 'rag_game2048_best'
const AI_INTERVAL = 220 // AI 托管每步间隔(ms)
const MAX_HISTORY = 10 // 撤销仅保留最近 10 步

// ---------- 棋盘逻辑 ----------
let tileSeq = 0

function emptyCells(board) {
  const cells = []
  for (let r = 0; r < SIZE; r++) {
    for (let c = 0; c < SIZE; c++) {
      if (!board[r][c]) cells.push([r, c])
    }
  }
  return cells
}

function toMatrix(tiles) {
  const m = Array.from({length: SIZE}, () => Array(SIZE).fill(0))
  for (const t of tiles) m[t.r][t.c] = t.value
  return m
}

function spawnTile(tiles) {
  const empty = emptyCells(toMatrix(tiles))
  if (!empty.length) return
  const [r, c] = empty[Math.floor(Math.random() * empty.length)]
  tiles.push({id: ++tileSeq, r, c, value: Math.random() < 0.1 ? 4 : 2, dying: false})
}

function newTiles() {
  const tiles = []
  spawnTile(tiles)
  spawnTile(tiles)
  return tiles
}

// 规划一次移动: 返回新矩阵 + 每个方块的去向(供渲染层平滑滑动), 移不动返回 null
function planMove(board, dir) {
  const next = board.map(row => row.slice())
  const moves = []
  let gained = 0
  for (let i = 0; i < SIZE; i++) {
    // slots: 按滑动目标方向排列(第一个 = 最先到达的位置)
    const slots = []
    for (let k = 0; k < SIZE; k++) {
      if (dir === 'L') slots.push([i, k])
      else if (dir === 'R') slots.push([i, SIZE - 1 - k])
      else if (dir === 'U') slots.push([k, i])
      else slots.push([SIZE - 1 - k, i])
    }
    const nums = slots
      .map(([r, c], idx) => ({v: board[r][c], idx}))
      .filter(x => x.v)
    const outs = []
    for (let n = 0; n < nums.length; n++) {
      if (n + 1 < nums.length && nums[n].v === nums[n + 1].v) {
        outs.push({v: nums[n].v * 2, srcs: [nums[n].idx, nums[n + 1].idx]})
        gained += nums[n].v * 2
        n++
      } else {
        outs.push({v: nums[n].v, srcs: [nums[n].idx]})
      }
    }
    for (let j = 0; j < SIZE; j++) {
      const [dr, dc] = slots[j]
      next[dr][dc] = outs[j] ? outs[j].v : 0
    }
    outs.forEach((out, j) => {
      const [toR, toC] = slots[j]
      out.srcs.forEach((srcIdx, si) => {
        const [fromR, fromC] = slots[srcIdx]
        moves.push({fromR, fromC, toR, toC, eaten: si > 0})
      })
    })
  }
  const changed = next.some((row, r) => row.some((v, c) => v !== board[r][c]))
  return changed ? {matrix: next, gained, moves} : null
}

function canMove(board) {
  if (emptyCells(board).length) return true
  for (let r = 0; r < SIZE; r++) {
    for (let c = 0; c < SIZE; c++) {
      const v = board[r][c]
      if (c + 1 < SIZE && board[r][c + 1] === v) return true
      if (r + 1 < SIZE && board[r + 1][c] === v) return true
    }
  }
  return false
}

// ---------- AI: expectimax ----------
// 算法对标 GitHub 最优开源实现: nneonneo/2048-AI(expectimax + 概率剪枝 + 置换表
// + 行表启发式)与 kcwu/2048-python(迭代加深 + 低概率方块剪枝)。
// 与 game2048.py(终端版)为同一算法, 仅时间预算/概率阈值按运行环境调参
// (浏览器算力远高于 CPython, 概率阈值取更低以搜得更深)。
// 与旧的固定深度+随机采样实现相比: 随机节点取全期望(无采样噪声)、深度随局面自适应
// (空格越少搜索越深)、置换表消除重复子树、时间预算内迭代加深。
const AI_TIME_BUDGET = 200 // 每步思考时间上限(ms), 时间内迭代加深
const AI_MAX_DEPTH = 8     // 迭代加深上限(己方步数)
const CPROB_THRESH = 1e-4  // 概率剪枝阈值: 累积概率低于该值的分支不再展开

// 启发式权重移植自 nneonneo/2048-AI: 空格、可合并对、单调性、方块总秩
const SCORE_LOST_PENALTY = 200000 // 存活棋盘基础分: 被将死的节点返回 0, 受强烈规避
const SCORE_MONOTONICITY_POWER = 4
const SCORE_MONOTONICITY_WEIGHT = 47
const SCORE_SUM_POWER = 3.5
const SCORE_SUM_WEIGHT = 11
const SCORE_MERGES_WEIGHT = 700
const SCORE_EMPTY_WEIGHT = 270

const TIMEOUT_CHECK = 512 // 每这么多节点检查一次剩余时间
const SPAWNS = [[1, 0.9], [2, 0.1]] // 新方块: 幂次与概率
const SEARCH_TIMEOUT = Symbol('timeout')

const quadKey = (a, b, c, d) => ((a * 17 + b) * 17 + c) * 17 + d

// 数值棋盘(0/2/4/...) -> 幂次棋盘(0=空, 1=2, 2=4, ...)
function toRanks(board) {
  const t = new Array(16)
  for (let i = 0; i < 16; i++) {
    const v = board[(i / 4) | 0][i % 4]
    t[i] = v ? 31 - Math.clz32(v) : 0
  }
  return t
}

// 行/列4元组(向左滑) -> [结果, 是否变化]
const slideCache = new Map()

function slideQuad(a, b, c, d) {
  const key = quadKey(a, b, c, d)
  let hit = slideCache.get(key)
  if (hit) return hit
  const nums = [a, b, c, d].filter(x => x)
  const out = []
  let i = 0
  while (i < nums.length) {
    if (i + 1 < nums.length && nums[i] === nums[i + 1]) {
      out.push(nums[i] + 1) // 幂次 +1 即数值翻倍
      i += 2
    } else {
      out.push(nums[i])
      i++
    }
  }
  while (out.length < 4) out.push(0)
  hit = [out, out[0] !== a || out[1] !== b || out[2] !== c || out[3] !== d]
  slideCache.set(key, hit)
  return hit
}

// 幂次棋盘上按 'L'/'R'/'U'/'D' 移动, 移不动返回 null
function moveRanks(t, dir) {
  const out = new Array(16)
  let changed = false
  const horiz = dir === 'L' || dir === 'R'
  const rev = dir === 'R' || dir === 'D'
  for (let i = 0; i < 4; i++) {
    const base = horiz ? i * 4 : i
    const step = horiz ? 1 : 4
    const a = t[base]
    const b = t[base + step]
    const c = t[base + 2 * step]
    const d = t[base + 3 * step]
    const [res, ch] = rev ? slideQuad(d, c, b, a) : slideQuad(a, b, c, d)
    changed = changed || ch
    for (let j = 0; j < 4; j++) {
      out[base + j * step] = res[rev ? 3 - j : j]
    }
  }
  return changed ? out : null
}

// 行/列4元组 -> 启发分
const heurCache = new Map()

function lineHeur(a, b, c, d) {
  const key = quadKey(a, b, c, d)
  const hit = heurCache.get(key)
  if (hit !== undefined) return hit
  const line = [a, b, c, d]
  let empty = 0
  let merges = 0
  let total = 0
  let prev = 0
  let counter = 0
  for (const rank of line) {
    total += rank ** SCORE_SUM_POWER
    if (rank === 0) {
      empty++
    } else {
      if (rank === prev) counter++
      else if (counter > 0) {
        merges += 1 + counter
        counter = 0
      }
      prev = rank
    }
  }
  if (counter > 0) merges += 1 + counter
  let monoLeft = 0
  let monoRight = 0
  for (let i = 1; i < 4; i++) {
    const x = line[i - 1] ** SCORE_MONOTONICITY_POWER
    const y = line[i] ** SCORE_MONOTONICITY_POWER
    if (line[i - 1] > line[i]) monoLeft += x - y
    else monoRight += y - x
  }
  const score = SCORE_LOST_PENALTY + SCORE_EMPTY_WEIGHT * empty + SCORE_MERGES_WEIGHT * merges
    - SCORE_MONOTONICITY_WEIGHT * Math.min(monoLeft, monoRight) - SCORE_SUM_WEIGHT * total
  heurCache.set(key, score)
  return score
}

function heur(t) {
  return lineHeur(t[0], t[1], t[2], t[3]) + lineHeur(t[4], t[5], t[6], t[7])
    + lineHeur(t[8], t[9], t[10], t[11]) + lineHeur(t[12], t[13], t[14], t[15])
    + lineHeur(t[0], t[4], t[8], t[12]) + lineHeur(t[1], t[5], t[9], t[13])
    + lineHeur(t[2], t[6], t[10], t[14]) + lineHeur(t[3], t[7], t[11], t[15])
}

// 一步棋的 expectimax 搜索: 概率剪枝 + 置换表 + 时间预算
class Searcher {
  constructor(deadline) {
    this.deadline = deadline
    this.nodes = 0
    this.table = new Map() // 幂次棋盘 -> [覆盖的剩余层数, 估值]
    this.hitLimit = false // 是否有节点被深度上限截断(继续加深才有意义)
  }

  tick() {
    if (++this.nodes >= TIMEOUT_CHECK) {
      this.nodes = 0
      if (Date.now() > this.deadline) throw SEARCH_TIMEOUT
    }
  }

  // 己方节点: 4 个方向取最大值; 无路可走(被将死)返回 0, 远低于存活估值
  player(t, cprob, depth, limit) {
    this.tick()
    let best = 0.0
    for (const d of ['L', 'R', 'U', 'D']) {
      const nb = moveRanks(t, d)
      if (nb) {
        const v = this.chance(nb, cprob, depth + 1, limit)
        if (v > best) best = v
      }
    }
    return best
  }

  // 随机节点: 对每个空格的 2/4 方块取期望; 低概率分支剪枝, 按保留概率归一
  chance(t, cprob, depth, limit) {
    this.tick()
    if (depth >= limit) {
      this.hitLimit = true
      return heur(t)
    }
    const covered = limit - depth
    const key = t.join(',')
    const entry = this.table.get(key)
    if (entry && entry[0] >= covered) return entry[1]
    const empty = []
    for (let i = 0; i < 16; i++) {
      if (!t[i]) empty.push(i)
    }
    const cp = cprob / empty.length
    if (cp * 0.9 < CPROB_THRESH) return heur(t) // 累积概率过低, 不再展开
    let total = 0
    for (const i of empty) {
      let cell = 0
      let kept = 0
      for (const [rank, p] of SPAWNS) {
        if (cp * p < CPROB_THRESH) continue // 低概率的 4 方块剪枝
        const t2 = t.slice()
        t2[i] = rank
        cell += p * this.player(t2, cp * p, depth, limit)
        kept += p
      }
      total += cell / kept
    }
    const res = total / empty.length
    this.table.set(key, [covered, res])
    return res
  }
}

// 返回最佳方向('L'/'R'/'U'/'D'), 无合法移动返回 null
function bestMove(board) {
  const t = toRanks(board)
  const moves = []
  for (const d of ['L', 'R', 'U', 'D']) {
    const nb = moveRanks(t, d)
    if (nb) moves.push([d, nb])
  }
  if (!moves.length) return null
  const deadline = Date.now() + AI_TIME_BUDGET
  let bestDir = moves[0][0] // 至少返回一个合法方向
  for (let limit = 2; limit <= AI_MAX_DEPTH; limit++) {
    const searcher = new Searcher(deadline)
    const values = {}
    let hitTimeout = false
    try {
      for (const [d, nb] of moves) {
        values[d] = searcher.chance(nb, 1.0, 0, limit)
      }
    } catch (e) {
      if (e !== SEARCH_TIMEOUT) throw e
      hitTimeout = true
    }
    if (hitTimeout) break
    bestDir = Object.entries(values).reduce((a, b) => (b[1] > a[1] ? b : a))[0]
    if (!searcher.hitLimit) break // 深度已被概率剪枝封顶, 继续加深结果不变
  }
  return bestDir
}

// ---------- Store ----------
export const useGame2048Store = defineStore('game2048', () => {
  const tiles = ref(newTiles())
  const score = ref(0)
  const best = ref(Number(localStorage.getItem(BEST_KEY)) || 0)
  const history = ref([])
  const aiEnabled = ref(false) // AI 托管开关
  const active = ref(false)    // 游戏弹窗是否打开(可见)
  const won = ref(false)
  let aiTimer = null

  const board = computed(() => toMatrix(tiles.value.filter(t => !t.dying)))
  const maxTile = computed(() => Math.max(...board.value.map(row => Math.max(...row))))
  const gameOver = computed(() => !canMove(board.value))

  function newGame() {
    tiles.value = newTiles()
    score.value = 0
    history.value = []
    won.value = false
    scheduleAi()
  }

  function move(dir) {
    const live = tiles.value.filter(t => !t.dying)
    const plan = planMove(toMatrix(live), dir)
    if (!plan) return false
    history.value = [...history.value.slice(-(MAX_HISTORY - 1)), {
      tiles: live.map(t => ({...t})),
      score: score.value,
    }]
    const byPos = new Map(live.map(t => [`${t.r},${t.c}`, t]))
    for (const mv of plan.moves) {
      const tile = byPos.get(`${mv.fromR},${mv.fromC}`)
      tile.r = mv.toR
      tile.c = mv.toC
      if (mv.eaten) tile.dying = true // 被合并的方块淡出后在下次操作时移除
      else tile.value = plan.matrix[mv.toR][mv.toC]
    }
    spawnTile(live)
    tiles.value = live
    score.value += plan.gained
    if (score.value > best.value) {
      best.value = score.value
      localStorage.setItem(BEST_KEY, String(best.value))
    }
    if (!won.value && maxTile.value >= 2048) won.value = true
    scheduleAi()
    return true
  }

  function undo() {
    const prev = history.value.pop()
    if (!prev) return false
    tiles.value = prev.tiles.map(t => ({...t}))
    score.value = prev.score
    won.value = maxTile.value >= 2048
    scheduleAi()
    return true
  }

  function setAiEnabled(enabled) {
    aiEnabled.value = enabled
    scheduleAi()
  }

  // 弹窗可见性: 打开时若开关仍开则继续托管; 关闭时强制关掉 AI 托管, 游戏状态保留
  function setActive(v) {
    active.value = v
    if (!v) aiEnabled.value = false // 关闭弹窗时必须关闭 AI 托管, 不留后台运行
    scheduleAi()
  }

  function aiStep() {
    if (!aiEnabled.value || !active.value || gameOver.value) return
    const dir = bestMove(board.value)
    if (dir) move(dir)
    else scheduleAi()
  }

  function scheduleAi() {
    if (aiTimer) {
      clearTimeout(aiTimer)
      aiTimer = null
    }
    if (aiEnabled.value && active.value && !gameOver.value) {
      aiTimer = setTimeout(aiStep, AI_INTERVAL)
    }
  }

  return {
    tiles,
    board,
    score,
    best,
    history,
    aiEnabled,
    active,
    won,
    maxTile,
    gameOver,
    newGame,
    move,
    undo,
    setAiEnabled,
    setActive,
  }
})
