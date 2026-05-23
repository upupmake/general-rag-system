import { siDeepseek, siGooglegemini, siBytedance, siMoonshotai, siMinimax, siAnthropic, siQwen, siXiaomi } from 'simple-icons';
import { writeFileSync, mkdirSync } from 'fs';

const dir = 'src/assets/providers';
mkdirSync(dir, { recursive: true });

const providers = {
  deepseek: { icon: siDeepseek },
  gemini: { icon: siGooglegemini },
  bytedance: { icon: siBytedance },
  moonshotai: { icon: siMoonshotai },
  minimax: { icon: siMinimax },
  anthropic: { icon: siAnthropic },
  qwen: { icon: siQwen },
  xiaomi: { icon: siXiaomi },
};

for (const [name, { icon }] of Object.entries(providers)) {
  const svg = `<svg role="img" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="${icon.path}" fill="#${icon.hex}"/></svg>`;
  writeFileSync(`${dir}/${name}.svg`, svg);
  console.log(`Created: ${name}.svg (hex: ${icon.hex})`);
}

console.log('Done generating simple-icons SVGs');