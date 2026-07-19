import markdownit from 'markdown-it';
import taskLists from 'markdown-it-task-lists';
import hljs from "highlight.js";
import 'highlight.js/styles/atom-one-light.css';

const langAliases = {
  'js': 'javascript',
  'ts': 'typescript',
  'py': 'python',
  'h5': 'html',
  'rb': 'ruby',
  'sh': 'bash',
  'c++': 'cpp',
};

const md = markdownit({
  html: true,
  linkify: true,
  typographer: true,
  highlight: function (str, lang) {
    lang = langAliases[lang] || lang;
    if (lang && hljs.getLanguage(lang)) {
      try {
        return '<pre class="hljs"><code>'
            + hljs.highlight(str, {language: lang, ignoreIllegals: true}).value +
            '</code></pre>';
      } catch (__) {}
    }
    const result = hljs.highlightAuto(str);
    return '<pre class="hljs"><code>' + result.value + '</code></pre>';
  }
}).use(taskLists);

export default md;
