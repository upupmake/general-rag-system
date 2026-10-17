import {existsSync, readFileSync, readdirSync, statSync} from 'node:fs'
import {extname, join, relative, resolve} from 'node:path'
import {spawnSync} from 'node:child_process'
import OSS from 'ali-oss'

const projectRoot = resolve(import.meta.dirname, '..')
const distDir = join(projectRoot, 'dist')
const envFile = join(projectRoot, '.env.deploy')
const args = new Set(process.argv.slice(2))
const dryRun = args.has('--dry-run')
const skipBuild = args.has('--skip-build')
const deleteStale = args.has('--delete')

const usage = `用法: npm run deploy:oss [-- 选项]

选项:
  --skip-build   跳过构建，直接上传现有 dist
  --dry-run      只列出将上传的文件，不连接 OSS
  --delete       上传后删除 OSS 前缀下不在 dist 中的旧文件
  --help         显示帮助

配置读取顺序: 系统环境变量优先，其次 rag-client/.env.deploy。`

if (args.has('--help')) {
    console.log(usage)
    process.exit(0)
}

const loadEnvFile = file => {
    if (!existsSync(file)) return
    for (const raw of readFileSync(file, 'utf8').split(/\r?\n/)) {
        const line = raw.trim()
        if (!line || line.startsWith('#')) continue
        const separator = line.indexOf('=')
        if (separator === -1) continue
        const key = line.slice(0, separator).trim()
        let value = line.slice(separator + 1).trim()
        if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
            value = value.slice(1, -1)
        }
        if (process.env[key] === undefined) process.env[key] = value
    }
}
loadEnvFile(envFile)

const required = ['OSS_REGION', 'OSS_BUCKET', 'OSS_ACCESS_KEY_ID', 'OSS_ACCESS_KEY_SECRET']
const missing = required.filter(key => !process.env[key])
if (missing.length) {
    console.error(`缺少 OSS 配置: ${missing.join(', ')}`)
    console.error('请复制 .env.deploy.example 为 .env.deploy 并填写，或通过环境变量提供。')
    process.exit(1)
}

const prefix = (process.env.OSS_PREFIX || '').replace(/^\/+|\/+$/g, '')
const endpoint = process.env.OSS_ENDPOINT || undefined
const keyFor = relativePath => (prefix ? `${prefix}/${relativePath}` : relativePath)

if (deleteStale && !prefix && !args.has('--allow-root-delete')) {
    console.error('OSS_PREFIX 为空时禁止 --delete，避免删除整个 Bucket；如确认要清理根目录，请加 --allow-root-delete。')
    process.exit(1)
}

if (!skipBuild) {
    console.log('构建前端...')
    const build = spawnSync('npm run build', {cwd: projectRoot, stdio: 'inherit', shell: true})
    if (build.status !== 0) process.exit(build.status ?? 1)
}

if (!existsSync(join(distDir, 'index.html'))) {
    console.error(`未找到 ${join(distDir, 'index.html')}，请先执行 npm run build。`)
    process.exit(1)
}

const walk = directory => readdirSync(directory).flatMap(name => {
    const absolute = join(directory, name)
    return statSync(absolute).isDirectory() ? walk(absolute) : [absolute]
})

const files = walk(distDir).map(absolute => ({
    absolute,
    relative: relative(distDir, absolute).replaceAll('\\', '/')
})).sort((a, b) => a.relative.localeCompare(b.relative))

const contentTypes = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.svg': 'image/svg+xml',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
    '.ico': 'image/x-icon',
    '.woff': 'font/woff',
    '.woff2': 'font/woff2',
    '.ttf': 'font/ttf',
    '.eot': 'application/vnd.ms-fontobject',
    '.map': 'application/json',
    '.txt': 'text/plain; charset=utf-8',
    '.wasm': 'application/wasm',
    '.mp4': 'video/mp4',
    '.webm': 'video/webm'
}

const headersFor = relativePath => ({
    'Content-Type': contentTypes[extname(relativePath).toLowerCase()] || 'application/octet-stream',
    'Cache-Control': relativePath === 'index.html'
        ? 'no-cache'
        : relativePath.startsWith('assets/')
            ? 'public, max-age=31536000, immutable'
            : 'public, max-age=3600'
})

console.log(`OSS: ${process.env.OSS_REGION} / ${process.env.OSS_BUCKET}${prefix ? ` / ${prefix}` : ''}`)
console.log(`文件数: ${files.length}${dryRun ? '（dry-run，不上传）' : ''}`)

if (dryRun) {
    for (const file of files.slice(0, 20)) {
        const key = keyFor(file.relative)
        const {['Cache-Control']: cache, ['Content-Type']: type} = headersFor(file.relative)
        console.log(`${key}  ${type}  ${cache}`)
    }
    if (files.length > 20) console.log(`... 其余 ${files.length - 20} 个文件`)
    process.exit(0)
}

const client = new OSS({
    region: process.env.OSS_REGION,
    bucket: process.env.OSS_BUCKET,
    accessKeyId: process.env.OSS_ACCESS_KEY_ID,
    accessKeySecret: process.env.OSS_ACCESS_KEY_SECRET,
    ...(endpoint ? {endpoint} : {}),
    secure: process.env.OSS_SECURE !== 'false'
})

const concurrency = Math.max(1, Number(process.env.OSS_CONCURRENCY || 8))
let cursor = 0
let uploaded = 0
const failures = []

const worker = async () => {
    while (cursor < files.length) {
        const file = files[cursor++]
        const key = keyFor(file.relative)
        try {
            await client.put(key, file.absolute, {headers: headersFor(file.relative)})
            uploaded++
            if (uploaded % 25 === 0 || uploaded === files.length) {
                console.log(`已上传 ${uploaded}/${files.length}`)
            }
        } catch (error) {
            failures.push({key, message: error.message})
        }
    }
}

await Promise.all(Array.from({length: Math.min(concurrency, files.length)}, worker))

if (failures.length) {
    console.error(`上传失败 ${failures.length} 个文件:`)
    failures.forEach(failure => console.error(`  ${failure.key}: ${failure.message}`))
    process.exit(1)
}

const indexHead = await client.head(keyFor('index.html'))
console.log(`验证 index.html: HTTP ${indexHead.status}, ${indexHead.res.headers['content-length']} bytes`)

if (deleteStale) {
    const localKeys = new Set(files.map(file => keyFor(file.relative)))
    let marker
    let deleted = 0
    do {
        const result = await client.list({prefix: prefix ? `${prefix}/` : '', marker, 'max-keys': 1000})
        for (const object of result.objects || []) {
            if (!localKeys.has(object.name)) {
                await client.delete(object.name)
                deleted++
            }
        }
        marker = result.isTruncated ? result.nextMarker : null
    } while (marker)
    console.log(`已删除 ${deleted} 个旧文件`)
}

console.log(`发布完成: ${uploaded} 个文件`)
