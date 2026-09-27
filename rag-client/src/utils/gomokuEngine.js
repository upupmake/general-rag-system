/**
 * 五子棋引擎（gomoku.py 的 JS 移植）
 *
 * 搜索特性：
 *  - PVS + 迭代加深
 *  - Zobrist 置换表
 *  - 杀手着法 / 历史启发
 *  - 线段窗口威胁分类棋型评估（五/活四/四/活三/眠三/二）
 *  - VCF 连续冲四威胁搜索（进攻 + 防守瓦解）
 *  - 强制着法剪枝（成五 / 防五）
 */

export const N = 15
export const N2 = N * N
export const EMPTY = 0
export const BLACK = 1
export const WHITE = 2

const DIRS = [
  [0, 1],
  [1, 0],
  [1, 1],
  [1, -1],
]

const WIN = 10 ** 9
const INF = 2 * WIN

// 棋型权重（评估用，远小于 WIN）
const W_FIVE = 10 ** 7
const W_OPEN_FOUR = 3 * 10 ** 6
const W_FOUR = 3 * 10 ** 5
const W_OPEN_THREE = 5 * 10 ** 4
const W_THREE = 8 * 10 ** 3
const W_DOUBLE_THREE = 4 * 10 ** 4
const W_OPEN_TWO = 1500
const W_TWO = 300
const W_ONE = 30

const VCF_MAX_PLY = 22

const now = () =>
  typeof performance !== 'undefined' ? performance.now() : Date.now()

// 落点形状分表 [连子数][封堵数]（着法排序用）
const SHAPE_SCORE = []
for (let cnt = 0; cnt < 9; cnt++) {
  const row = [0, 0, 0]
  for (let bl = 0; bl < 3; bl++) {
    let s
    if (cnt >= 5) s = 10 ** 6
    else if (cnt === 4) s = bl === 0 ? 10 ** 5 : 10 ** 4
    else if (cnt === 3) s = bl === 0 ? 1000 : 100
    else if (cnt === 2) s = bl === 0 ? 100 : 10
    else s = 10
    row[bl] = s
  }
  SHAPE_SCORE.push(row)
}

// ---------------------------------------------------------------- 棋盘几何
function buildLines() {
  const lines = []
  for (let r = 0; r < N; r++) {
    const line = []
    for (let c = 0; c < N; c++) line.push(r * N + c)
    lines.push(line)
  }
  for (let c = 0; c < N; c++) {
    const line = []
    for (let r = 0; r < N; r++) line.push(r * N + c)
    lines.push(line)
  }
  for (const [dr, dc] of [
    [1, 1],
    [1, -1],
  ]) {
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const pr = r - dr
        const pc = c - dc
        if (pr >= 0 && pr < N && pc >= 0 && pc < N) continue // 只取每条线起点
        const line = []
        let rr = r
        let cc = c
        while (rr >= 0 && rr < N && cc >= 0 && cc < N) {
          line.push(rr * N + cc)
          rr += dr
          cc += dc
        }
        if (line.length >= 5) lines.push(line)
      }
    }
  }
  return lines
}

const LINES = buildLines()

const WINDOWS = []
for (const line of LINES) {
  for (let i = 0; i + 5 <= line.length; i++) {
    WINDOWS.push(line.slice(i, i + 5))
  }
}

const CELL_LINES = []
for (let i = 0; i < N2; i++) CELL_LINES.push([])
LINES.forEach((line, li) => {
  line.forEach((cell, pos) => {
    CELL_LINES[cell].push([li, pos])
  })
})

const NEIGH_OFF = []
for (let r = 0; r < N; r++) {
  for (let c = 0; c < N; c++) {
    const lst = []
    for (let dr = -2; dr <= 2; dr++) {
      for (let dc = -2; dc <= 2; dc++) {
        const rr = r + dr
        const cc = c + dc
        if (rr >= 0 && rr < N && cc >= 0 && cc < N) lst.push(rr * N + cc)
      }
    }
    NEIGH_OFF.push(lst)
  }
}

// ---------------------------------------------------------------- 棋型评估
const SEG_CACHE = new Map()

function windowStats(seg) {
  const wm = []
  const we = []
  for (let k = 0; k + 5 <= seg.length; k++) {
    let m = 0
    const es = []
    for (let t = k; t < k + 5; t++) {
      if (seg[t] === '1') m++
      else es.push(t)
    }
    wm.push(m)
    we.push(es)
  }
  return [wm, we]
}

/**
 * 线段（只含 '0' 空 / '1' 己方，两端天然被对方或边界阻塞）的形状分。
 * 威胁阶梯：五(已连五) / 四(已有成五点) / 三(一步成四) / 二(一步成四点) / 一
 */
export function analyzeSegment(seg) {
  const hit = SEG_CACHE.get(seg)
  if (hit !== undefined) return hit

  const L = seg.length
  let ones = 0
  for (let i = 0; i < L; i++) if (seg[i] === '1') ones++
  if (L < 5 || ones < 2) {
    SEG_CACHE.set(seg, 0)
    return 0
  }

  const [wm, we] = windowStats(seg)
  const nw = L - 4

  // 五（已连五）
  for (let k = 0; k < nw; k++) {
    if (wm[k] === 5) {
      SEG_CACHE.set(seg, W_FIVE)
      return W_FIVE
    }
  }

  let score = 0

  // 四：当前已存在的成五点
  const comp = new Set()
  for (let k = 0; k < nw; k++) {
    if (wm[k] === 4 && we[k].length === 1) comp.add(we[k][0])
  }
  if (comp.size > 0) {
    score += comp.size === 1 ? W_FOUR : W_OPEN_FOUR + (comp.size - 1) * W_FOUR
  }

  // 三：落 i 后的成五点集合（= 一步成四）
  const threes = new Map()
  for (let k = 0; k < nw; k++) {
    if (wm[k] === 3 && we[k].length === 2) {
      const [a, b] = we[k]
      if (!threes.has(a)) threes.set(a, new Set())
      threes.get(a).add(b)
      if (!threes.has(b)) threes.set(b, new Set())
      threes.get(b).add(a)
    }
  }
  let openThree = 0
  let closeThree = 0
  for (const s of threes.values()) {
    if (s.size >= 2) openThree++
    else closeThree++
  }
  score += openThree * W_OPEN_THREE + closeThree * W_THREE
  if (openThree >= 2) score += W_DOUBLE_THREE

  // 二：落 i 后的成四点集合
  let openTwo = 0
  let closeTwo = 0
  for (let i = 0; i < L; i++) {
    if (seg[i] !== '0') continue
    const s = new Set()
    for (let k = 0; k < nw; k++) {
      const es = we[k]
      if (wm[k] === 3 && es.length === 2 && es[0] !== i && es[1] !== i) {
        s.add(es[0])
        s.add(es[1])
      } else if (wm[k] === 2 && es.length === 3 && (es[0] === i || es[1] === i || es[2] === i)) {
        for (const x of es) if (x !== i) s.add(x)
      }
    }
    if (s.size >= 2) openTwo++
    else if (s.size === 1) closeTwo++
  }
  score += openTwo * W_OPEN_TWO + closeTwo * W_TWO

  // 一：两子潜力窗口
  let ones5 = 0
  for (let k = 0; k < nw; k++) {
    if (wm[k] === 2 && we[k].length === 3) ones5++
  }
  score += ones5 * W_ONE

  SEG_CACHE.set(seg, score)
  return score
}

/** 落子到 cell 后是否成五（cell 视为 side 的棋子）。 */
export function makesFive(board, cell, side) {
  const r = (cell / N) | 0
  const c = cell % N
  for (const [dr, dc] of DIRS) {
    let cnt = 1
    for (const sgn of [1, -1]) {
      let rr = r + dr * sgn
      let cc = c + dc * sgn
      while (rr >= 0 && rr < N && cc >= 0 && cc < N && board[rr * N + cc] === side) {
        cnt++
        rr += dr * sgn
        cc += dc * sgn
      }
    }
    if (cnt >= 5) return true
  }
  return false
}

/** 落子后判胜（供 store 使用）。 */
export function checkWin(board, cell, side) {
  return makesFive(board, cell, side)
}

export function fmtMove(cell) {
  const r = (cell / N) | 0
  const c = cell % N
  return String.fromCharCode(65 + c) + r
}

// ---------------------------------------------------------------- 引擎
export class GomokuEngine {
  constructor(timeLimit = 1.0, maxDepth = 12, useVcf = true) {
    this.timeLimit = timeLimit // 秒
    this.maxDepth = maxDepth
    this.useVcf = useVcf
    this.zobHi = [new Uint32Array(N2), new Uint32Array(N2)]
    this.zobLo = [new Uint32Array(N2), new Uint32Array(N2)]
    for (let s = 0; s < 2; s++) {
      for (let i = 0; i < N2; i++) {
        this.zobHi[s][i] = Math.floor(Math.random() * 0xffffffff) >>> 0
        this.zobLo[s][i] = Math.floor(Math.random() * 0xffffffff) >>> 0
      }
    }
    this.reset()
  }

  // ---------------- 棋盘维护 ----------------
  reset() {
    this.board = new Int8Array(N2)
    this.stones = 0
    this.stoneList = []
    this.keyHi = 0
    this.keyLo = 0
    this.side = BLACK
    this.total = [0, 0, 0]
    this.lineScores = []
    for (let i = 0; i < LINES.length; i++) this.lineScores.push([0, 0, 0])
    this.tt = new Map()
    this.killers = []
    for (let i = 0; i < 64; i++) this.killers.push([0, 0])
    this.history = [new Int32Array(N2), new Int32Array(N2)]
    this.nodes = 0
    this.aborted = false
  }

  makeMove(cell, side) {
    this.board[cell] = side
    this.keyHi ^= this.zobHi[side - 1][cell]
    this.keyLo ^= this.zobLo[side - 1][cell]
    this.stoneList.push(cell)
    this.stones++
    this.refreshLines(cell)
  }

  unmakeMove(cell, side) {
    this.board[cell] = EMPTY
    this.keyHi ^= this.zobHi[side - 1][cell]
    this.keyLo ^= this.zobLo[side - 1][cell]
    this.stoneList.pop()
    this.stones--
    this.refreshLines(cell)
  }

  refreshLines(cell) {
    for (const [li] of CELL_LINES[cell]) {
      const ls = this.lineScores[li]
      const old1 = ls[1]
      const old2 = ls[2]
      const new1 = this.lineScore(li, 1)
      const new2 = this.lineScore(li, 2)
      ls[1] = new1
      ls[2] = new2
      this.total[1] += new1 - old1
      this.total[2] += new2 - old2
    }
  }

  lineScore(li, color) {
    const board = this.board
    let total = 0
    let seg = ''
    let ones = 0
    for (const cell of LINES[li]) {
      const v = board[cell]
      if (v === color) {
        seg += '1'
        ones++
      } else if (v === EMPTY) {
        seg += '0'
      } else if (seg !== '') {
        if (ones >= 2) total += analyzeSegment(seg)
        seg = ''
        ones = 0
      }
    }
    if (seg !== '' && ones >= 2) total += analyzeSegment(seg)
    return total
  }

  evaluate() {
    const me = this.side
    return this.total[me] - this.total[3 - me]
  }

  // ---------------- 走法生成 ----------------
  genCandidates() {
    if (this.stones === 0) return [7 * N + 7]
    const cand = new Set()
    for (const cell of this.stoneList) {
      for (const idx of NEIGH_OFF[cell]) cand.add(idx)
    }
    const out = []
    for (const c of cand) if (this.board[c] === EMPTY) out.push(c)
    return out
  }

  threatMoves(side) {
    const five = new Set()
    const four = new Map()
    const board = this.board
    for (const w of WINDOWS) {
      let m = 0
      const es = []
      let ok = true
      for (const cell of w) {
        const v = board[cell]
        if (v === side) m++
        else if (v === EMPTY) es.push(cell)
        else {
          ok = false
          break
        }
      }
      if (!ok) continue
      if (m === 4 && es.length === 1) {
        five.add(es[0])
      } else if (m === 3 && es.length === 2) {
        const [a, b] = es
        if (!four.has(a)) four.set(a, new Set())
        four.get(a).add(b)
        if (!four.has(b)) four.set(b, new Set())
        four.get(b).add(a)
      }
    }
    return [five, four]
  }

  /** 返回 (攻分, 防分, 成五掩码)；掩码 bit1=落 me 成五, bit2=落 opp 成五。 */
  moveInfo(cell, me) {
    const r = (cell / N) | 0
    const c = cell % N
    const board = this.board
    const opp = 3 - me
    let tm = 0
    let to = 0
    let mask = 0
    for (const [dr, dc] of DIRS) {
      let cm = 1
      let co = 1
      let bm = 0
      let bo = 0
      for (const sgn of [1, -1]) {
        let rr = r + dr * sgn
        let cc = c + dc * sgn
        const v = rr >= 0 && rr < N && cc >= 0 && cc < N ? board[rr * N + cc] : 3
        if (v === me) {
          while (rr >= 0 && rr < N && cc >= 0 && cc < N && board[rr * N + cc] === me) {
            cm++
            rr += dr * sgn
            cc += dc * sgn
          }
          if (!(rr >= 0 && rr < N && cc >= 0 && cc < N) || board[rr * N + cc] !== EMPTY) bm++
          bo++
        } else if (v === opp) {
          while (rr >= 0 && rr < N && cc >= 0 && cc < N && board[rr * N + cc] === opp) {
            co++
            rr += dr * sgn
            cc += dc * sgn
          }
          if (!(rr >= 0 && rr < N && cc >= 0 && cc < N) || board[rr * N + cc] !== EMPTY) bo++
          bm++
        } else if (v !== EMPTY) {
          bm++
          bo++
        }
      }
      tm += SHAPE_SCORE[cm < 8 ? cm : 8][bm]
      to += SHAPE_SCORE[co < 8 ? co : 8][bo]
      if (cm >= 5) mask |= 1
      if (co >= 5) mask |= 2
    }
    return [tm, to, mask]
  }

  scanMoves(cands, me) {
    const myFive = []
    const oppFive = []
    const pm = new Map()
    const po = new Map()
    for (const c of cands) {
      const [a, b, m] = this.moveInfo(c, me)
      pm.set(c, a)
      po.set(c, b)
      if (m & 1) myFive.push(c)
      else if (m & 2) oppFive.push(c)
    }
    return [myFive, oppFive, pm, po]
  }

  orderMoves(cands, pm, po, ply, ttMove) {
    const me = this.side
    const k0 = this.killers[ply][0]
    const k1 = this.killers[ply][1]
    const hist = this.history[me - 1]
    const scored = []
    for (const c of cands) {
      let s = pm.get(c) * 10 + po.get(c) * 9
      if (ttMove && c === ttMove) s += 10 ** 9
      else if (c === k0) s += 8 * 10 ** 6
      else if (c === k1) s += 7 * 10 ** 6
      s += hist[c]
      scored.push([s, c])
    }
    scored.sort((x, y) => y[0] - x[0] || x[1] - y[1])
    return scored.map((x) => x[1])
  }

  // ---------------- 搜索 ----------------
  checkTime() {
    if ((this.nodes & 1023) === 0 && now() > this.deadline) this.aborted = true
  }

  negamax(depth, alpha, beta, ply) {
    this.nodes++
    this.checkTime()
    if (this.aborted) return 0
    if (this.stones === N2) return 0

    const me = this.side
    const opp = 3 - me
    const alpha0 = alpha

    // 置换表探查
    let ttMove = 0
    const key = this.keyHi * 4294967296 + this.keyLo
    const entry = this.tt.get(key)
    if (entry) {
      const [d, flag, sc0, mv] = entry
      ttMove = mv
      if (d >= depth) {
        let s = sc0
        if (s > WIN / 2) s -= ply
        else if (s < -WIN / 2) s += ply
        if (flag === 0) return s
        if (flag === 1 && s >= beta) return s
        if (flag === 2 && s <= alpha) return s
      }
    }

    if (depth <= 0) return this.evaluate()

    let cands = this.genCandidates()
    const [myFive, oppFive, pm, po] = this.scanMoves(cands, me)
    if (myFive.length > 0) return WIN - ply
    if (oppFive.length > 0) {
      const blocks = new Set(oppFive)
      if (blocks.size === 1) cands = [...blocks] // 唯一防点，强制
      else return -(WIN - ply) // 双五无法兼顾
    }

    const moves = this.orderMoves(cands, pm, po, ply, ttMove)

    let best = -INF
    let bestMove = 0
    for (let i = 0; i < moves.length; i++) {
      const cell = moves[i]
      this.makeMove(cell, me)
      this.side = opp
      let sc
      if (i === 0) {
        sc = -this.negamax(depth - 1, -beta, -alpha, ply + 1)
      } else {
        sc = -this.negamax(depth - 1, -alpha - 1, -alpha, ply + 1)
        if (sc > alpha && sc < beta) {
          sc = -this.negamax(depth - 1, -beta, -alpha, ply + 1)
        }
      }
      this.side = me
      this.unmakeMove(cell, me)
      if (this.aborted) return 0
      if (sc > best) {
        best = sc
        bestMove = cell
        if (sc > alpha) {
          alpha = sc
          if (alpha >= beta) {
            const k = this.killers[ply]
            if (k[0] !== cell) {
              k[1] = k[0]
              k[0] = cell
            }
            this.history[me - 1][cell] += depth * depth
            break
          }
        }
      }
    }

    // 置换表保存（胜负分做 ply 修正）
    let flag
    if (best <= alpha0) flag = 2
    else if (best >= beta) flag = 1
    else flag = 0
    let s = best
    if (s > WIN / 2) s += ply
    else if (s < -WIN / 2) s -= ply
    if (this.tt.size > 1500000) this.tt.clear()
    this.tt.set(key, [depth, flag, s, bestMove])
    return best
  }

  rootSearch(depth) {
    const me = this.side
    let cands = this.genCandidates()
    const [myFive, oppFive, pm, po] = this.scanMoves(cands, me)
    if (myFive.length > 0) return [WIN, myFive[0]]
    if (oppFive.length > 0) {
      const blocks = new Set(oppFive)
      return [0, [...blocks][0]]
    }
    const moves = this.orderMoves(cands, pm, po, 0, 0)

    let best = -INF
    let bestMove = moves[0]
    let alpha = -INF
    for (let i = 0; i < moves.length; i++) {
      const cell = moves[i]
      this.makeMove(cell, me)
      this.side = 3 - me
      let sc
      if (i === 0) {
        sc = -this.negamax(depth - 1, -INF, INF, 1)
      } else {
        sc = -this.negamax(depth - 1, -alpha - 1, -alpha, 1)
        if (sc > alpha) sc = -this.negamax(depth - 1, -INF, -alpha, 1)
      }
      this.side = me
      this.unmakeMove(cell, me)
      if (this.aborted) return [best, bestMove]
      if (sc > best) {
        best = sc
        bestMove = cell
        if (sc > alpha) alpha = sc
      }
    }
    return [best, bestMove]
  }

  // ---------------- VCF 威胁搜索 ----------------
  vcf(side, ply, deadline) {
    if (ply > VCF_MAX_PLY || now() > deadline) return null
    const opp = 3 - side
    const [five, four] = this.threatMoves(side)
    if (five.size > 0) return [...five][0]
    const [oFive] = this.threatMoves(opp)
    if (oFive.size > 0) return null // 对方抢先成五，本线进攻失败

    const moves = [...four.keys()].sort(
      (a, b) => four.get(b).size - four.get(a).size
    )
    for (const c of moves) {
      this.makeMove(c, side)
      const [five2] = this.threatMoves(side)
      if (five2.size >= 2) {
        this.unmakeMove(c, side)
        return c // 双四/四四，无法防守
      }
      if (five2.size === 1) {
        const t = [...five2][0]
        const [oFive2, oFour2] = this.threatMoves(opp)
        let ok = true
        if (oFive2.size > 0) {
          ok = false // 对方可反击成五
        } else {
          const responses = new Set([t, ...oFour2.keys()])
          for (const r of responses) {
            this.makeMove(r, opp)
            if (makesFive(this.board, r, opp)) {
              this.unmakeMove(r, opp)
              ok = false
              break
            }
            const res = this.vcf(side, ply + 2, deadline)
            this.unmakeMove(r, opp)
            if (res === null) {
              ok = false
              break
            }
          }
        }
        this.unmakeMove(c, side)
        if (ok) return c
      } else {
        this.unmakeMove(c, side)
      }
    }
    return null
  }

  vcfAttack(side) {
    return this.vcf(side, 0, now() + Math.min(0.5 * this.timeLimit * 1000, 1500))
  }

  vcfDefense(side) {
    const opp = 3 - side
    const deadline = now() + Math.min(0.3 * this.timeLimit * 1000, 1000)
    if (this.vcf(opp, 0, deadline) === null) return null
    const cands = this.genCandidates()
    const [, , pm, po] = this.scanMoves(cands, side)
    const moves = this.orderMoves(cands, pm, po, 0, 0).slice(0, 12)
    for (const c of moves) {
      this.makeMove(c, side)
      const res = this.vcf(opp, 0, deadline)
      this.unmakeMove(c, side)
      if (res === null) return c
    }
    return null
  }

  // ---------------- 入口 ----------------
  search() {
    const t0 = now()
    this.deadline = t0 + this.timeLimit * 1000
    this.nodes = 0
    this.aborted = false

    const me = this.side
    const cands = this.genCandidates()
    const [myFive, oppFive] = this.scanMoves(cands, me)

    if (myFive.length > 0) return this.result(myFive[0], WIN, 0, t0, '成五')
    if (oppFive.length > 0) {
      const blocks = [...new Set(oppFive)].sort((a, b) => a - b)
      return this.result(blocks[0], 0, 0, t0, '防五')
    }

    if (this.useVcf) {
      const mv = this.vcfAttack(me)
      if (mv !== null) return this.result(mv, WIN / 2, 0, t0, 'VCF 取胜')
      const mv2 = this.vcfDefense(me)
      if (mv2 !== null) return this.result(mv2, 0, 0, t0, '瓦解对方 VCF')
    }

    let bestMove = null
    let bestScore = 0
    let reached = 0
    for (let depth = 1; depth <= this.maxDepth; depth++) {
      this.aborted = false
      const [score, move] = this.rootSearch(depth)
      if (this.aborted) break
      bestMove = move
      bestScore = score
      reached = depth
      if (Math.abs(bestScore) > WIN / 2) break
    }
    if (bestMove === null) {
      const [, , pm, po] = this.scanMoves(cands, me)
      bestMove = this.orderMoves(cands, pm, po, 0, 0)[0]
    }
    return this.result(bestMove, bestScore, reached, t0, '')
  }

  result(move, score, depth, t0, reason) {
    return {
      move,
      score,
      depth,
      nodes: this.nodes,
      time: (now() - t0) / 1000,
      reason,
    }
  }
}
