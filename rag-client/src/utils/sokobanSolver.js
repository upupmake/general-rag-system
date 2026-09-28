// 推箱子求解器（JavaScript 版, 与 Python 版 sokoban.py 同源算法）
//
// 关卡出处: Microban 题集, 作者 David W. Skinner（2000 年发布, 2000-04 修订）,
//   公开授权自由分发使用。
// 已知最优推动数对照: Sokoban Solver Statistics - Push-Optimal Test Suite
//   （Sokolution 7.32, 2025-04-11）:
//   https://sokoban-solver-statistics.sourceforge.io/statistics/Push-OptimalTestSuite/Sokolution%20-%20Microban.html
//
// 算法（与 sokoban.py 一致）:
//   状态 = 箱子位置集合 + 玩家可达区域（按区域最小格规范化合并, 可哈希去重）;
//   两阶段 anytime A*: 加权 A*(w=2) 快速拿种子解, 再无权 A* 证明/改进最优,
//   全程受时间预算约束, 超时优雅返回当前最好结果（尽力而为）;
//   启发式 = 箱子到目标的最小代价二分图匹配（箱子数 <= 5 用子集 DP 精确求解）;
//   死锁剪枝 = 死格（逆向拉箱子可达域, 覆盖角落/贴墙无目标等简单死锁）
//     + 2x2 冻结死锁 + 密封直隧道宏动作（一次推到底, 只停隧道目标格,
//     每次推动单独计数, 不损失 push 最优性; 无目标的密封隧道判死）。
//
// 对外 API（纯函数, 无 DOM/Worker 依赖, 可直接被 node import 自测）:
//   DIRS                                   方向常量 {U:[0,-1], D:[0,1], L:[-1,0], R:[1,0]}
//   getLevels()                            全部关卡摘要 [{index,title,w,h,boxCount,optimalPushes}]
//   getLevel(index)                        1 起的关卡对象（纯数据, 可 structured clone 进 Worker）
//   isSolved(level, boxes)                 是否通关（所有箱子在目标上）
//   tryMove(level, boxes, player, dir)     走/推一步 → {boxes, player, pushed} 或 null
//   solveLevel(level, boxes, player, budgetMs)  求解 → {pushes, moves, pushCount, ...}
//
// 坐标约定: 公开 API 的位置均为 [x, y] 数组（x 列, y 行）; boxes 为 [x,y] 数组列表。

// 方向: 与解法串 UDLR 约定一致
export const DIRS = {U: [0, -1], D: [0, 1], L: [-1, 0], R: [1, 0]}
const DIR_LIST = [DIRS.U, DIRS.D, DIRS.L, DIRS.R]
const CH_OF_DIR = {'0,-1': 'U', '0,1': 'D', '-1,0': 'L', '1,0': 'R'}

const STRIDE = 32            // 内部坐标编码步长: enc = y * STRIDE + x（棋盘 <= 16x16 足够）
const INF = 1 << 28

const now = typeof performance !== 'undefined' ? () => performance.now() : () => Date.now()

class SearchTimeout extends Error {}

// ---------- 内嵌关卡 ----------
// 关卡出处: Microban 题集, 作者 David W. Skinner (2000), 公开授权自由分发。
// 格式: '#' 墙, ' ' 地板, '$' 箱子, '.' 目标, '*' 箱子在目标上,
//       '@' 玩家, '+' 玩家在目标上。关卡之间用 '---' 行/注释行/空行分隔。
const LEVELS_RAW = `; Microban #2
######
#    #
# #@ #
# $* #
# .* #
#    #
######

---
; Microban #21
####
#  ####
# . . #
# $$#@#
##    #
 ######

---
; Microban #5
 #######
 #     #
 # .$. #
## $@$ #
#  .$. #
#      #
########

---
; Microban #4
########
#      #
# .**$@#
#      #
#####  #
    ####

---
; Microban #1
####
# .#
#  ###
#*@  #
#  $ #
#  ###
####

---
; Microban #17
#####
# @ #
#...#
#$$$##
#    #
#    #
######

---
; Microban #9
#####
#.  ##
#@$$ #
##   #
 ##  #
  ##.#
   ###

---
; Microban #14
#######
#     #
# # # #
#. $*@#
#   ###
#####

---
; Microban #12
#####
#   ##
# $  #
## $ ####
 ###@.  #
  #  .# #
  #     #
  #######

---
; Microban #15
     ###
######@##
#    .* #
#   #   #
#####$# #
    #   #
    #####

---
; Microban #18
#######
#     #
#. .  #
# ## ##
#  $ #
###$ #
  #@ #
  #  #
  ####

---
; Microban #3
  ####
###  ####
#     $ #
# #  #$ #
# . .#@ #
#########

---
; Microban #11
  ######
  #    #
  # ##@##
### # $ #
# ..# $ #
#       #
#  ######
####

---
; Microban #20
#######
#     ###
#  @$$..#
#### ## #
  #     #
  #  ####
  #  #
  ####

---
; Microban #13
####
#. ##
#.@ #
#. $#
##$ ###
 # $  #
 #    #
 #  ###
 ####`

// 各关卡已知最优推动数（与关卡顺序一致）。
// 来源: Sokoban Solver Statistics - Push-Optimal Test Suite, Sokolution 7.32（2025-04-11）。
const OPTIMAL_PUSHES = [3, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 13, 16, 16, 21]

// ---------- 关卡解析 ----------
function parseLevels() {
  const levels = []
  let title = ''
  let rows = []
  const flush = () => {
    if (rows.length) {
      const idx = levels.length + 1
      levels.push(buildLevel(idx, title || `关卡 ${idx}`, rows))
      rows = []
    }
  }
  for (const raw of LEVELS_RAW.split('\n')) {
    const line = raw.replace(/\s+$/, '')
    if (line.startsWith(';')) {
      title = line.replace(/^;+\s*/, '').trim()
    } else if (line.trim() === '---' || line.trim() === '') {
      flush()
      title = ''
    } else {
      rows.push(line)
    }
  }
  flush()
  return levels
}

function buildLevel(index, title, rows) {
  const h = rows.length
  let w = 0
  for (const r of rows) w = Math.max(w, r.length)
  const walls = []
  const floors = []
  const goals = []
  const startBoxes = []
  let startPlayer = null
  for (let y = 0; y < h; y++) {
    const row = rows[y]
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]
      const c = [x, y]
      if (ch === '#') {
        walls.push(c)
      } else if (' .*$@+'.includes(ch)) {
        floors.push(c)
        if (ch === '.' || ch === '*') goals.push(c)
        if (ch === '$' || ch === '*') startBoxes.push(c)
        if (ch === '@' || ch === '+') startPlayer = c
      }
      // 其余字符视作棋盘外的虚空
    }
  }
  if (!startPlayer) throw new Error(`关卡 ${index}: 缺少玩家 '@'`)
  if (startBoxes.length !== goals.length) throw new Error(`关卡 ${index}: 箱子数与目标数不一致`)
  return {
    index,
    title,
    w,
    h,
    optimalPushes: OPTIMAL_PUSHES[index - 1] ?? null,
    walls,
    floors,
    goals,
    startBoxes,
    startPlayer,
  }
}

const LEVELS = parseLevels()

export function getLevels() {
  return LEVELS.map(l => ({
    index: l.index,
    title: l.title,
    w: l.w,
    h: l.h,
    boxCount: l.startBoxes.length,
    optimalPushes: l.optimalPushes,
  }))
}

export function getLevel(index) {
  const lv = LEVELS[index - 1]
  if (!lv) throw new Error(`关卡编号无效: ${index}`)
  return lv
}

// ---------- 公共规则 ----------
export function isSolved(level, boxes) {
  const goals = new Set(level.goals.map(c => c[0] + ',' + c[1]))
  for (const [x, y] of boxes) if (!goals.has(x + ',' + y)) return false
  return boxes.length > 0
}

// 走/推一步。返回 {boxes, player, pushed} 或 null（撞墙/推不动）。
// boxes 输出为规范化排序数组, 不修改入参。
export function tryMove(level, boxes, player, dir) {
  const d = DIRS[dir]
  if (!d) return null
  const inn = intern(level)
  const boxSet = new Set(boxes.map(([x, y]) => y * STRIDE + x))
  const px = player[0] + d[0]
  const py = player[1] + d[1]
  const nxt = py * STRIDE + px
  if (!inn.floorSet.has(nxt)) return null
  if (boxSet.has(nxt)) {
    const bx = px + d[0]
    const by = py + d[1]
    const beyond = by * STRIDE + bx
    if (!inn.floorSet.has(beyond) || boxSet.has(beyond)) return null
    boxSet.delete(nxt)
    boxSet.add(beyond)
    return {boxes: sortedCells(boxSet), player: [px, py], pushed: true}
  }
  return {boxes: sortedCells(boxSet), player: [px, py], pushed: false}
}

function sortedCells(encSet) {
  return [...encSet].sort((a, b) => a - b).map(e => [e % STRIDE, (e / STRIDE) | 0])
}

// ---------- 内部静态分析（按 level 对象缓存） ----------
const internCache = new WeakMap()

function intern(level) {
  let inn = internCache.get(level)
  if (inn) return inn
  const floorSet = new Set(level.floors.map(([x, y]) => y * STRIDE + x))
  const wallSet = new Set(level.walls.map(([x, y]) => y * STRIDE + x))
  const goalList = level.goals.map(([x, y]) => y * STRIDE + x)
  const goalSet = new Set(goalList)
  // 死格表: 逆向"拉箱子"可达域, dist 无穷大即死格（玩家假设可瞬移, 只会高估可达性, 判死可靠）
  const pullDist = new Map()
  for (const c of floorSet) pullDist.set(c, INF)
  const dq = []
  for (const g of goalList) {
    pullDist.set(g, 0)
    dq.push(g)
  }
  for (let head = 0; head < dq.length; head++) {
    const x = dq[head]
    const x0 = x % STRIDE
    const y0 = (x / STRIDE) | 0
    for (const [dx, dy] of DIR_LIST) {
      // 箱子在 y 格, 朝 (dx,dy) 推一下落到 x; 玩家需站在 y-d
      const yx = x0 - dx
      const yy = y0 - dy
      const y = yy * STRIDE + yx
      const stand = (yy - dy) * STRIDE + (yx - dx)
      if (floorSet.has(y) && floorSet.has(stand) && pullDist.get(y) >= INF) {
        pullDist.set(y, pullDist.get(x) + 1)
        dq.push(y)
      }
    }
  }
  const deadSet = new Set()
  for (const c of floorSet) if (pullDist.get(c) >= INF) deadSet.add(c)
  // 各目标出发的走路距离（忽略箱子, 只避墙）, 供匹配启发式用
  const goalWalk = goalList.map(g => walkDist(floorSet, g))
  inn = {floorSet, wallSet, goalList, goalSet, deadSet, goalWalk}
  internCache.set(level, inn)
  return inn
}

function walkDist(floorSet, start) {
  const dist = new Int32Array(STRIDE * STRIDE).fill(INF)
  dist[start] = 0
  const dq = [start]
  for (let head = 0; head < dq.length; head++) {
    const x = dq[head]
    const x0 = x % STRIDE
    const y0 = (x / STRIDE) | 0
    for (const [dx, dy] of DIR_LIST) {
      const y = (y0 + dy) * STRIDE + (x0 + dx)
      if (floorSet.has(y) && dist[y] >= INF) {
        dist[y] = dist[x] + 1
        dq.push(y)
      }
    }
  }
  return dist
}

// ---------- 死锁检测 ----------
// 刚放到 moved 的箱子是否形成"墙+箱子"全满的 2x2 死块, 且块里有未进目标的箱子
// （块内箱子永远推不动 => 死锁）。棋盘外坐标按墙处理。
function freeze2x2(inn, boxSet, moved) {
  const mx = moved % STRIDE
  const my = (moved / STRIDE) | 0
  for (const ox of [-1, 0]) {
    for (const oy of [-1, 0]) {
      const cells = []
      for (const i of [0, 1]) for (const j of [0, 1]) cells.push((my + oy + j) * STRIDE + (mx + ox + i))
      let full = true
      let bad = false
      for (const c of cells) {
        if (inn.floorSet.has(c) && !boxSet.has(c)) full = false
        if (boxSet.has(c) && !inn.goalSet.has(c)) bad = true
      }
      if (full && bad) return true
    }
  }
  return false
}

// 生成把 box 朝 d 推的后继局面: [{boxSet, rec, playerEnd}]
// 含隧道宏动作: 密封直隧道（两侧是墙, 尽头是墙）里一次推到底,
// 只在隧道内目标格停步; 无目标的密封隧道直接判死。
function expandPushes(inn, boxSet, box, d) {
  const dx = d[0]
  const dy = d[1]
  const px = -dy
  const py = dx
  const bx = box % STRIDE
  const by = (box / STRIDE) | 0
  const first = (by + dy) * STRIDE + (bx + dx)
  if (!inn.floorSet.has(first) || boxSet.has(first)) return []
  if (inn.deadSet.has(first)) return [] // 落到死格: 剪枝

  // ---- 探测密封直隧道 ----
  const path = [first]
  let cur = first
  let sealed = false
  for (;;) {
    const cx = cur % STRIDE
    const cy = (cur / STRIDE) | 0
    if (inn.floorSet.has((cy + py) * STRIDE + (cx + px)) ||
        inn.floorSet.has((cy - py) * STRIDE + (cx - px))) {
      break // 侧向开口: 不是密封隧道
    }
    const nx = cx + dx
    const ny = cy + dy
    const nxt = ny * STRIDE + nx
    if (boxSet.has(nxt)) break // 尽头是箱子, 可能从另一头绕进来, 不宏
    if (!inn.floorSet.has(nxt)) {
      sealed = true // 尽头是墙: 盒子进去后只能一路推到底
      break
    }
    if (inn.deadSet.has(nxt)) break // 前方是死格但可穿行, 不敢断言单调
    path.push(nxt)
    cur = nxt
  }

  const out = []
  if (sealed) {
    for (let j = 0; j < path.length; j++) {
      const cell = path[j]
      if (!inn.goalSet.has(cell)) continue // 隧道内只允许停在目标格
      const nb = new Set(boxSet)
      nb.delete(box)
      nb.add(cell)
      if (freeze2x2(inn, nb, cell)) continue
      const rec = [[box, path[0]]]
      for (let k = 1; k <= j; k++) rec.push([path[k - 1], path[k]])
      out.push({boxSet: nb, rec, playerEnd: cell - dy * STRIDE - dx})
    }
    return out // 无隧道目标 => 进隧道即死锁, 全剪
  }

  // ---- 普通单推 ----
  const nb = new Set(boxSet)
  nb.delete(box)
  nb.add(first)
  if (freeze2x2(inn, nb, first)) return []
  return [{boxSet: nb, rec: [[box, first]], playerEnd: box}]
}

// ---------- 启发式 ----------
// 箱子到目标的最小代价二分图匹配（子集 DP 精确）, 代价 = 忽略其它箱子的推动距离, 可采纳。
function heuristic(inn, boxArr) {
  const n = boxArr.length
  const size = 1 << n
  let dp = new Array(size).fill(INF)
  dp[0] = 0
  for (let bi = 0; bi < n; bi++) {
    const b = boxArr[bi]
    const row = inn.goalWalk.map(d => d[b])
    const ndp = new Array(size).fill(INF)
    for (let mask = 0; mask < size; mask++) {
      const base = dp[mask]
      if (base >= INF) continue
      for (let gi = 0; gi < n; gi++) {
        if (mask & (1 << gi)) continue
        const nm = mask | (1 << gi)
        const c = base + row[gi]
        if (c < ndp[nm]) ndp[nm] = c
      }
    }
    dp = ndp
  }
  return dp[size - 1]
}

// ---------- 搜索 ----------
// 玩家从 start 出发（避开箱子）可达区域, 返回 {min, seen}（min 用作规范化代表）
function region(inn, boxSet, start) {
  const seen = new Set([start])
  const stack = [start]
  let min = start
  while (stack.length) {
    const x = stack.pop()
    if (x < min) min = x
    const x0 = x % STRIDE
    const y0 = (x / STRIDE) | 0
    for (const [dx, dy] of DIR_LIST) {
      const y = (y0 + dy) * STRIDE + (x0 + dx)
      if (inn.floorSet.has(y) && !boxSet.has(y) && !seen.has(y)) {
        seen.add(y)
        stack.push(y)
      }
    }
  }
  return {min, seen}
}

class MinHeap {
  constructor() {
    this.a = []
  }

  get size() {
    return this.a.length
  }

  push(item) {
    const a = this.a
    a.push(item)
    let i = a.length - 1
    while (i > 0) {
      const p = (i - 1) >> 1
      if (cmpTuple(a[p], a[i]) <= 0) break
      ;[a[p], a[i]] = [a[i], a[p]]
      i = p
    }
  }

  pop() {
    const a = this.a
    const top = a[0]
    const last = a.pop()
    if (a.length) {
      a[0] = last
      let i = 0
      for (;;) {
        const l = i * 2 + 1
        const r = l + 1
        let m = i
        if (l < a.length && cmpTuple(a[l], a[m]) < 0) m = l
        if (r < a.length && cmpTuple(a[r], a[m]) < 0) m = r
        if (m === i) break
        ;[a[m], a[i]] = [a[i], a[m]]
        i = m
      }
    }
    return top
  }
}

function cmpTuple(x, y) {
  for (let i = 0; i < x.length; i++) {
    if (x[i] !== y[i]) return x[i] < y[i] ? -1 : 1
  }
  return 0
}

function stateKey(boxArr, regionMin) {
  return boxArr.join(',') + '|' + regionMin
}

// 加权 A*（weight=1 为标准 A*）。返回推动记录 [[from,to],...] 或 null（耗尽=无解）。
// 超时抛 SearchTimeout。costCap: 剪掉 f >= costCap 的分支（用于证明种子解最优）。
function search(inn, boxArr0, player0, deadline, weight, costCap, stats) {
  const boxSet0 = new Set(boxArr0)
  const reg0 = region(inn, boxSet0, player0)
  const key0 = stateKey(boxArr0, reg0.min)
  const h0 = heuristic(inn, boxArr0)
  if (h0 >= INF) return null
  const gbest = new Map([[key0, 0]])
  const came = new Map()
  const heap = new MinHeap()
  heap.push([weight * h0, h0, 0, 0, key0, boxArr0, reg0.min])
  let seq = 1
  while (heap.size) {
    if (now() > deadline) throw new SearchTimeout()
    const [, , g, , key, boxArr, rmin] = heap.pop()
    if (gbest.get(key) !== g) continue // 过期堆条目
    stats.nodes++
    const boxSet = new Set(boxArr)
    if (isSolvedEnc(inn, boxSet)) {
      // 逆序回溯推动链
      const chain = []
      let k = key
      while (came.has(k)) {
        const {parent, rec} = came.get(k)
        chain.push(rec)
        k = parent
      }
      chain.reverse()
      return chain.flat()
    }
    const reg = region(inn, boxSet, rmin)
    for (const box of boxArr) {
      const bx = box % STRIDE
      const by = (box / STRIDE) | 0
      for (const d of DIR_LIST) {
        const behind = (by - d[1]) * STRIDE + (bx - d[0])
        if (!reg.seen.has(behind)) continue
        for (const {boxSet: nb, rec, playerEnd} of expandPushes(inn, boxSet, box, d)) {
          const ng = g + rec.length
          if (costCap !== null && ng >= costCap) continue
          const nbArr = [...nb].sort((a, b) => a - b)
          const nh = heuristic(inn, nbArr)
          if (nh >= INF) continue
          if (costCap !== null && ng + nh >= costCap) continue
          const nreg = region(inn, nb, playerEnd)
          const nkey = stateKey(nbArr, nreg.min)
          if ((gbest.get(nkey) ?? INF) <= ng) continue // 只保留更优 g（reopening）
          gbest.set(nkey, ng)
          came.set(nkey, {parent: key, rec})
          heap.push([ng + weight * nh, nh, ng, seq++, nkey, nbArr, nreg.min])
        }
      }
    }
  }
  return null
}

function isSolvedEnc(inn, boxSet) {
  for (const b of boxSet) if (!inn.goalSet.has(b)) return false
  return true
}

// ---------- 解法展开 ----------
// 玩家从 src 走到 dst（避开箱子与墙）的最短移动串; 不可达返回 null
function walkPath(inn, boxSet, src, dst) {
  if (src === dst) return ''
  const prev = new Map([[src, null]])
  const dq = [src]
  for (let head = 0; head < dq.length; head++) {
    const cur = dq[head]
    const cx = cur % STRIDE
    const cy = (cur / STRIDE) | 0
    for (const d of DIR_LIST) {
      const y = (cy + d[1]) * STRIDE + (cx + d[0])
      if (inn.floorSet.has(y) && !boxSet.has(y) && !prev.has(y)) {
        prev.set(y, [cur, CH_OF_DIR[d[0] + ',' + d[1]]])
        if (y === dst) {
          const out = []
          let c = y
          while (prev.get(c)) {
            const [p, ch] = prev.get(c)
            out.push(ch)
            c = p
          }
          return out.reverse().join('')
        }
        dq.push(y)
      }
    }
  }
  return null
}

// 把推动记录展开成完整 UDLR 移动串（含推动前的走路）
function pushesToMoves(inn, boxArr0, player0, pushes) {
  const boxSet = new Set(boxArr0)
  let player = player0
  const out = []
  for (const [src, dst] of pushes) {
    const d = [dst % STRIDE - (src % STRIDE), ((dst / STRIDE) | 0) - ((src / STRIDE) | 0)]
    const ch = CH_OF_DIR[d[0] + ',' + d[1]]
    const stand = src - d[1] * STRIDE - d[0]
    const walk = walkPath(inn, boxSet, player, stand)
    if (walk === null) return '' // 正常搜索结果不会发生
    out.push(walk, ch)
    boxSet.delete(src)
    boxSet.add(dst)
    player = src
  }
  return out.join('')
}

// ---------- 求解入口 ----------
// 两阶段 anytime: 加权 A*(w=2) 找种子解, 无权 A* 证明/改进最优。
// 返回 {pushes, moves, pushCount, moveCount, nodes, elapsedMs, optimal, solvable, timedOut}
export function solveLevel(level, boxes, player, budgetMs = 30000) {
  const t0 = now()
  const budget = Math.max(0, budgetMs)
  const inn = intern(level)
  const boxArr0 = boxes.map(([x, y]) => y * STRIDE + x).sort((a, b) => a - b)
  const player0 = player[1] * STRIDE + player[0]
  const stats = {nodes: 0}
  const res = {
    pushes: null,
    moves: '',
    pushCount: 0,
    moveCount: 0,
    nodes: 0,
    elapsedMs: 0,
    optimal: false,
    solvable: true,
    timedOut: false,
  }

  // 初始死锁快判
  const boxSet0 = new Set(boxArr0)
  if (boxArr0.some(b => inn.deadSet.has(b)) ||
      boxArr0.some(b => freeze2x2(inn, boxSet0, b))) {
    res.solvable = false
    res.elapsedMs = now() - t0
    return res
  }

  let best = null

  // ---- 阶段 1: 加权 A* 找种子解 ----
  const slice1 = Math.min(3000, budget * 0.3)
  try {
    const got = search(inn, boxArr0, player0, t0 + slice1, 2.0, null, stats)
    if (got) best = got
    else res.solvable = false // 搜索耗尽: 已证明无解
  } catch (e) {
    if (!(e instanceof SearchTimeout)) throw e
  }

  // ---- 阶段 2: 无权 A* 证明/改进最优 ----
  if (res.solvable) {
    try {
      const got = search(inn, boxArr0, player0, t0 + budget, 1.0, best ? best.length : null, stats)
      if (got) {
        best = got
        res.optimal = true
      } else if (best) {
        res.optimal = true // 搜尽且无更好解: 种子解即最优
      } else {
        res.solvable = false
      }
    } catch (e) {
      if (!(e instanceof SearchTimeout)) throw e
      res.timedOut = true // 超时: 报告当前最好结果
    }
  }

  res.nodes = stats.nodes
  res.elapsedMs = now() - t0
  if (best) {
    res.pushes = best.map(([src, dst]) => [
      [src % STRIDE, (src / STRIDE) | 0],
      [dst % STRIDE, (dst / STRIDE) | 0],
    ])
    res.pushCount = best.length
    res.moves = pushesToMoves(inn, boxArr0, player0, best)
    res.moveCount = res.moves.length
  }
  return res
}
