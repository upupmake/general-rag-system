/* 五子棋 AI Web Worker：在后台线程搜索，不阻塞 UI */
import { GomokuEngine } from '../utils/gomokuEngine.js'

self.onmessage = (e) => {
  const { id, moves, side, timeLimit, maxDepth } = e.data
  try {
    const engine = new GomokuEngine(timeLimit, maxDepth, true)
    for (const [cell, s] of moves) engine.makeMove(cell, s)
    engine.side = side
    const info = engine.search()
    self.postMessage({ id, ...info })
  } catch (err) {
    self.postMessage({ id, move: null, error: String(err) })
  }
}
