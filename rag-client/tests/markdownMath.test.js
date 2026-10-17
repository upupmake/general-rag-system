import assert from 'node:assert/strict'
import {describe, test} from 'node:test'
import MarkdownIt from 'markdown-it'
import taskLists from 'markdown-it-task-lists'
import markdownMath from '../src/utils/markdownMath.js'

for (const html of [false, true]) {
  describe(`Markdown math (html: ${html})`, () => {
    const md = new MarkdownIt({html, linkify: true, typographer: true})
      .use(taskLists).use(markdownMath)
    const mathTokens = (source) => md.parse(source, {})
      .flatMap(token => token.children || [token])
      .filter(token => token.type === 'math_inline' || token.type === 'math_block')

    for (const [name, source, content, type] of [
      ['inline parentheses', String.raw`策略 \(\pi(a\mid s)\)`, String.raw`\pi(a\mid s)`, 'math_inline'],
      ['single-line brackets', String.raw`\[E = mc^2\]`, 'E = mc^2', 'math_block'],
      ['multiline brackets', String.raw`\[
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
\]`, String.raw`
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
`, 'math_block'],
      ['brackets in prose', String.raw`公式：\[x^2\]，结束。`, 'x^2', 'math_block'],
      ['parentheses in a heading', String.raw`## 价值 \(V^\pi(s)\)`, String.raw`V^\pi(s)`, 'math_inline'],
      ['parentheses in a table', String.raw`| 符号 |
| --- |
| \(\gamma\) |`, String.raw`\gamma`, 'math_inline'],
      ['parentheses in a link label', String.raw`[\(x^2\)](/docs)`, 'x^2', 'math_inline'],
      ['inline dollars', String.raw`折扣 $\gamma = 0.9$`, String.raw`\gamma = 0.9`, 'math_inline'],
      ['block dollars', String.raw`$$E = mc^2$$`, 'E = mc^2\n', 'math_block'],
    ]) {
      test(name, () => {
        const tokens = mathTokens(source)
        assert.equal(tokens.length, 1)
        assert.equal(tokens[0].content, content)
        assert.equal(tokens[0].type, type)
        const rendered = md.render(source)
        assert.match(rendered, /<svg\b/)
        assert.doesNotMatch(rendered, /<g\b[^>]*data-mml-node="merror"/)
        if (type === 'math_block') {
          assert.match(rendered, /<mjx-container\b[^>]*display="true"/)
        } else {
          assert.doesNotMatch(rendered, /<mjx-container\b[^>]*display="true"/)
        }
      })
    }

    test('preserves LaTeX commands, subscripts and line breaks', () => {
      const content = String.raw`
\begin{aligned}
G_t &= r_{t+1} + \gamma V^\pi(s) \\
x &= \text{a\_b} + \frac{1}{2}
\end{aligned}
`
      const source = '\\[' + content + '\\]'
      const [token] = mathTokens(source)
      assert.equal(token?.content, content)
      assert.doesNotMatch(md.render(source), /<g\b[^>]*data-mml-node="merror"/)
    })

    test('interrupts paragraphs without consuming surrounding text', () => {
      const tokens = md.parse('前文\n\\[\nx^2\n\\]\n后文', {})
      assert.deepEqual(tokens.map(token => token.type), [
        'paragraph_open', 'inline', 'paragraph_close', 'math_block',
        'paragraph_open', 'inline', 'paragraph_close',
      ])
      assert.deepEqual(tokens[3].map, [1, 4])
      assert.equal(tokens[1].content, '前文')
      assert.equal(tokens[5].content, '后文')
    })

    for (const [name, source] of [
      ['blockquote', '> \\[\n> \\gamma^2\n> \\]'],
      ['list', '- \\[\n  \\gamma^2\n  \\]\n- 下一项'],
      ['indented non-code block', '   \\[\n   \\gamma^2\n   \\]'],
    ]) {
      test(`display math in ${name}`, () => {
        const tokens = mathTokens(source)
        assert.equal(tokens.length, 1)
        assert.equal(tokens[0].type, 'math_block')
        assert.equal(tokens[0].content.trim(), String.raw`\gamma^2`)
        assert.match(md.render(source), /<svg\b/)
      })
    }

    test('keeps trailing prose after a bracket formula', () => {
      const source = String.raw`\[x^2\] 后文`
      assert.equal(mathTokens(source).length, 1)
      assert.match(md.render(source), /后文/)
    })

    test('ignores an escaped closing delimiter', () => {
      const [token] = mathTokens(String.raw`\(a\\) + b\)`)
      assert.equal(token?.content, String.raw`a\\) + b`)
    })

    const code = String.raw`\(x_1\) + \[\gamma^2\] + $a$ + $$b$$`
    for (const [name, source, type, expected] of [
      ['inline code', '`' + code + '`', 'code_inline', code],
      ['double-backtick code', '`` ' + code + ' `literal` ``', 'code_inline', code + ' `literal`'],
      ['backtick fence', '```latex\n' + code + '\n```', 'fence', code + '\n'],
      ['tilde fence', '~~~latex\n' + code + '\n~~~', 'fence', code + '\n'],
      ['indented code', '    ' + code, 'code_block', code + '\n'],
      ['tab-indented code', '\t' + code, 'code_block', code + '\n'],
    ]) {
      test(`does not parse or rewrite ${name}`, () => {
        assert.equal(mathTokens(source).length, 0)
        const tokens = md.parse(source, {}).flatMap(token => token.children || [token])
        assert.equal(tokens.find(token => token.type === type)?.content, expected)
        assert.doesNotMatch(md.render(source), /<svg\b/)
      })
    }

    for (const [name, source] of [
      ['escaped parentheses', String.raw`\\(x\\)`],
      ['escaped brackets', String.raw`\\[x\\]`],
      ['unclosed parentheses', String.raw`\(x + 1`],
      ['unclosed brackets', '\\[\nx + 1\n\n后文'],
      ['mismatched delimiters', '\\(x\\]\n\n\\[y\\)'],
      ['ordinary punctuation', '[text] (text)'],
    ]) {
      test(`leaves ${name} as ordinary Markdown`, () => {
        assert.equal(mathTokens(source).length, 0)
        assert.doesNotMatch(md.render(source), /<svg\b/)
      })
    }

    test('renders only completed formulas during streaming', () => {
      const complete = String.raw`\[\gamma^2\]`
      assert.equal(mathTokens(complete.slice(0, -1)).length, 0)
      assert.equal(mathTokens(complete).length, 1)
    })

    test('does not change the Markdown source', () => {
      const source = String.raw`行内 \(x_1\)，块级 \[\gamma^2\]，代码 ` + '`' + code + '`'
      const state = new md.core.State(source, md, {})
      md.core.process(state)
      assert.equal(state.src, source)
    })
  })
}
