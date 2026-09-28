// 推箱子 AI 求解 Worker: 复用 utils/sokobanSolver.js 的 anytime A*
// （与 Python 版 sokoban.py 同源算法）, 在后台线程计算, 不阻塞 UI。
// 返回 moves = 完整 UDLR 移动串（含推动前的走路）, 供托管逐步回放。
import {solveLevel} from '../utils/sokobanSolver.js'

self.onmessage = (e) => {
  const {id, level, boxes, player, budgetMs} = e.data
  const res = solveLevel(level, boxes, player, budgetMs ?? 30000)
  self.postMessage({
    id,
    moves: res.moves,
    pushCount: res.pushCount,
    moveCount: res.moveCount,
    nodes: res.nodes,
    elapsedMs: res.elapsedMs,
    optimal: res.optimal,
    solvable: res.solvable,
    timedOut: res.timedOut,
  })
}
