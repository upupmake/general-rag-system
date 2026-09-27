// 图片拼图 AI 求解 Worker: 复用 utils/puzzleSolver.js 的 IDA* 策略(与 Python 版 puzzle.py 同源),
// 在后台线程计算, 不阻塞 UI。返回值 moves = 依次移入空位的方块编号。
import {solve} from '../utils/puzzleSolver.js'

self.onmessage = (e) => {
  const {id, state, n} = e.data
  // 3x3 恒求最优; 4x4 先试最优(节点预算内), 超预算退化为快速近似(对应 Python 版 ε-IDA* 快速模式)
  let moves = solve(state, n, {weight: 1, nodeBudget: n === 3 ? Infinity : 3000000})
  if (!moves && n === 4) moves = solve(state, n, {weight: 2, nodeBudget: 3000000})
  self.postMessage({id, moves})
}
