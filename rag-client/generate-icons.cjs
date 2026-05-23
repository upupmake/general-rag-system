const si = require('simple-icons');
const fs = require('fs');
const dir = 'src/assets/providers';
if (!fs.existsSync(dir)) { fs.mkdirSync(dir, { recursive: true }); }

const providers = {
  deepseek: { key: 'siDeepseek', hex: '5786FE' },
  gemini: { key: 'siGooglegemini', hex: '8E75B2' },
  bytedance: { key: 'siBytedance', hex: '3C8CFF' },
  moonshotai: { key: 'siMoonshotai', hex: '000000' },
  minimax: { key: 'siMinimax', hex: 'E73562' },
  anthropic: { key: 'siAnthropic', hex: '191919' },
  qwen: { key: 'siQwen', hex: '6950EF' },
  xiaomi: { key: 'siXiaomi', hex: 'FF6900' }
};

for (const [name, { key, hex }] of Object.entries(providers)) {
  const icon = si[key];
  const svgContent = `<svg role="img" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="${icon.path}" fill="#${hex}"/></svg>`;
  fs.writeFileSync(`${dir}/${name}.svg`, svgContent);
  console.log(`Created: ${name}.svg (hex: ${hex})`);
}
console.log('Done generating 8 simple-icons SVGs');