import mathjax from 'markdown-it-mathjax3'

function findClosingDelimiter(source, start, end, closing) {
  for (let pos = start; pos + 1 < end; pos++) {
    if (source[pos] !== '\\') continue
    if (source[pos + 1] === closing) return pos
    pos++ // 成对反斜杠属于公式内容，不能作为结束定界符。
  }
  return -1
}

export default function markdownMath(md) {
  md.use(mathjax)

  // 在 Markdown 转义规则之前生成公式 token，不改写原文，也不进入代码 token。
  md.inline.ruler.before('escape', 'math_bracket_inline', (state, silent) => {
    const start = state.pos
    const opening = state.src.slice(start, start + 2)
    if (start + 1 >= state.posMax || (opening !== '\\(' && opening !== '\\[')) return false

    const closing = opening === '\\(' ? ')' : ']'
    const end = findClosingDelimiter(state.src, start + 2, state.posMax, closing)
    if (end === -1) return false

    if (!silent) {
      const type = opening === '\\(' ? 'math_inline' : 'math_block'
      const token = state.push(type, 'math', 0)
      token.content = state.src.slice(start + 2, end)
      token.markup = opening
    }
    state.pos = end + 2
    return true
  })

  md.block.ruler.after('blockquote', 'math_bracket_block', (state, startLine, endLine, silent) => {
    if (state.sCount[startLine] - state.blkIndent >= 4) return false
    const start = state.bMarks[startLine] + state.tShift[startLine]
    if (state.src.slice(start, start + 2) !== '\\[') return false

    for (let line = startLine; line < endLine; line++) {
      if (line > startLine && !state.isEmpty(line) && state.sCount[line] < state.blkIndent) return false
      const lineStart = state.bMarks[line] + state.tShift[line]
      const lineEnd = state.eMarks[line]
      const end = findClosingDelimiter(state.src, line === startLine ? start + 2 : lineStart, lineEnd, ']')
      if (end === -1) continue
      // 同行还有正文时，交给行内规则解析，避免吞掉公式后的文字。
      if (state.src.slice(end + 2, lineEnd).trim()) return false
      if (silent) return true

      const token = state.push('math_block', 'math', 0)
      token.block = true
      token.markup = '\\['
      token.map = [startLine, line + 1]
      token.content = line === startLine
        ? state.src.slice(start + 2, end)
        : state.src.slice(start + 2, state.eMarks[startLine]) + '\n'
          + state.getLines(startLine + 1, line, state.blkIndent, true)
          + state.src.slice(lineStart, end)
      state.line = line + 1
      return true
    }
    return false
  }, {alt: ['paragraph', 'reference', 'blockquote', 'list']})
}
