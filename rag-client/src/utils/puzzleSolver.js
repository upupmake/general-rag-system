// 图片拼图求解器 —— 与 D:\PythonProject\randomProject\puzzle.py 同策略:
// IDA* + 曼哈顿距离/线性冲突启发式。3x3 恒最优(weight=1);
// 4x4 默认快速近似(weight=2, 对应 Python 版 ε-IDA* 快速模式), weight=1 返回最优解。
// 返回值: 移动的方块编号数组(依次移入空位); 已复原返回 []; 不可解或超预算返回 null。

export const BLANK = 0 // 空位内部表示(界面上显示为缺块)

const neighborCache = new Map()

export function neighborTable(n) {
  if (!neighborCache.has(n)) {
    const table = []
    for (let i = 0; i < n * n; i++) {
      const r = Math.floor(i / n)
      const c = i % n
      const cells = []
      if (r > 0) cells.push(i - n)
      if (r < n - 1) cells.push(i + n)
      if (c > 0) cells.push(i - 1)
      if (c < n - 1) cells.push(i + 1)
      table.push(cells)
    }
    neighborCache.set(n, table)
  }
  return neighborCache.get(n)
}

export function isSolved(state) {
  for (let i = 0; i < state.length - 1; i++) if (state[i] !== i + 1) return false
  return state[state.length - 1] === BLANK
}

// 目标局面空位在右下角: 置换奇偶性 + 空位到右下的曼哈顿距离 之和为偶数才可解
export function isSolvable(state, n) {
  const perm = state.map(v => (v === BLANK ? n * n : v))
  let inv = 0
  for (let i = 0; i < perm.length; i++) {
    for (let j = i + 1; j < perm.length; j++) if (perm[i] > perm[j]) inv++
  }
  const bi = state.indexOf(BLANK)
  const dist = (n - 1 - Math.floor(bi / n)) + (n - 1 - (bi % n))
  return (inv + dist) % 2 === 0
}

// 随机打乱 1..n*n-1, 空位固定在最后一格; 不可解时交换前两个方块翻转奇偶性
export function shuffled(n, rng = Math.random) {
  const tiles = []
  for (let v = 1; v < n * n; v++) tiles.push(v)
  for (let i = tiles.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    const t = tiles[i]
    tiles[i] = tiles[j]
    tiles[j] = t
  }
  if (!isSolvable([...tiles, BLANK], n)) {
    const t = tiles[0]
    tiles[0] = tiles[1]
    tiles[1] = t
  }
  return [...tiles, BLANK]
}

// ---------- 曼哈顿 + 线性冲突 ----------
function makeHeuristic(n) {
  const goalPos = [null]
  for (let v = 1; v < n * n; v++) goalPos.push([Math.floor((v - 1) / n), (v - 1) % n])
  return function h(board) {
    let total = 0
    for (let idx = 0; idx < board.length; idx++) {
      const v = board[idx]
      if (v === BLANK) continue
      const gp = goalPos[v]
      total += Math.abs(gp[0] - Math.floor(idx / n)) + Math.abs(gp[1] - (idx % n))
    }
    // 线性冲突: 同一行(列)上目标同行(列)但顺序相反的方块对, 各加 2
    for (let r = 0; r < n; r++) {
      const base = r * n
      for (let i = 0; i < n; i++) {
        const vi = board[base + i]
        if (vi === BLANK || goalPos[vi][0] !== r) continue
        for (let j = i + 1; j < n; j++) {
          const vj = board[base + j]
          if (vj !== BLANK && goalPos[vj][0] === r && goalPos[vi][1] > goalPos[vj][1]) total += 2
        }
      }
    }
    for (let c = 0; c < n; c++) {
      for (let i = 0; i < n; i++) {
        const vi = board[i * n + c]
        if (vi === BLANK || goalPos[vi][1] !== c) continue
        for (let j = i + 1; j < n; j++) {
          const vj = board[j * n + c]
          if (vj !== BLANK && goalPos[vj][1] === c && goalPos[vi][0] > goalPos[vj][0]) total += 2
        }
      }
    }
    return total
  }
}

// ---------- IDA* ----------
const FOUND = Symbol('found')
const ABORT = Symbol('abort')

function search(ctx, board, g, hval, bound, path, lastTile) {
  const {nb, h, weight} = ctx
  const f = g + hval * weight
  if (f > bound) return f
  if (isSolved(board)) return FOUND
  if (++ctx.nodes >= ctx.nodeBudget) return ABORT
  const bi = board.indexOf(BLANK)
  const cands = []
  for (const q of nb[bi]) {
    const tile = board[q]
    if (tile === lastTile) continue // 不走回上一步
    board[bi] = tile
    board[q] = BLANK
    const ch = h(board)
    cands.push([g + 1 + ch * weight, ch, q, tile])
    board[q] = tile
    board[bi] = BLANK
  }
  cands.sort((a, b) => a[0] - b[0])
  let best = Infinity
  for (const [fc, ch, q, tile] of cands) {
    if (fc > bound) {
      if (fc < best) best = fc
      continue
    }
    board[bi] = tile
    board[q] = BLANK
    path.push(tile)
    const t = search(ctx, board, g + 1, ch, bound, path, tile)
    if (t === FOUND) return FOUND
    if (t === ABORT) return ABORT
    if (t < best) best = t
    path.pop()
    board[q] = tile
    board[bi] = BLANK
  }
  return best
}

export function solve(state, n, opts = {}) {
  const {weight = 1, nodeBudget = Infinity} = opts
  if (!isSolvable(state, n)) return null
  if (isSolved(state)) return []
  const board = state.slice()
  const h = makeHeuristic(n)
  const hval = h(board)
  const ctx = {nb: neighborTable(n), h, weight, nodes: 0, nodeBudget}
  const path = []
  let bound = hval * weight
  for (;;) {
    const t = search(ctx, board, 0, hval, bound, path, 0)
    if (t === FOUND) return path.slice()
    if (t === ABORT || t === Infinity) return null
    bound = t
  }
}
