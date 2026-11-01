# Contributing to General RAG System

感谢您对 General RAG System 项目的关注！我们欢迎任何形式的贡献。

## 📋 目录

- [行为准则](#行为准则)
- [如何贡献](#如何贡献)
- [开发流程](#开发流程)
- [代码规范](#代码规范)
- [提交规范](#提交规范)
- [问题反馈](#问题反馈)

## 行为准则

参与本项目即表示您同意遵守我们的[行为准则](CODE_OF_CONDUCT.md)。请确保在所有互动中保持尊重和专业。

## 如何贡献

### 🐛 报告 Bug

如果您发现了 Bug，请：

1. 检查 [Issues](https://github.com/cockmake/general-rag-system/issues) 确认问题是否已被报告
2. 如果没有，创建新 Issue 并包含：
   - 清晰的标题和描述
   - 复现步骤
   - 预期行为 vs 实际行为
   - 系统环境信息（OS、版本等）
   - 相关日志或截图

### 💡 提出新功能

如果您有新功能建议：

1. 先在 [Issues](https://github.com/cockmake/general-rag-system/issues) 中搜索是否已有类似建议
2. 创建 Feature Request Issue 说明：
   - 功能的使用场景
   - 为什么需要这个功能
   - 可能的实现方案

### 🔧 提交代码

1. **Fork 项目**
   ```bash
   git clone https://github.com/YOUR_USERNAME/general-rag-system.git
   cd general-rag-system
   ```

2. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   # 或
   git checkout -b fix/your-bug-fix
   ```

3. **进行开发**
   - 遵循项目的代码规范
   - 添加必要的测试
   - 更新相关文档

4. **提交更改**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

5. **推送到 Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **创建 Pull Request**
   - 在 GitHub 上创建 PR
   - 填写 PR 模板
   - 关联相关 Issue

## 开发流程

### 环境搭建

参考 [README.md](README.md) 中的部署指南设置开发环境。

### 项目结构

```
general-rag-system/
├── rag-client/          # Vue 3 前端（端口: 5173）
├── rag-server/          # Spring Boot 后端（端口: 8080）
├── rag-llm/             # Python FastAPI LLM 服务（端口: 8848）
├── rag-mcp/             # FastMCP 服务（端口: 8858）
├── embedding_rerank/    # vLLM 本地 Embedding/Rerank 服务
├── general_rag_database.sql  # 数据库初始化 SQL
└── docs/                # 文档
```

### 运行测试

**前端测试**
```bash
cd rag-client
npm run build
node --test tests/*.test.js
```

**后端测试**
```powershell
cd rag-server
$env:JAVA_HOME = 'D:\JDK-11'
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
mvn -o test
```

**Python 测试**
```bash
cd rag-llm
pytest tests -q

cd ../rag-mcp
pytest tests -q
```

## 代码规范

### JavaScript/Vue

- 遵循 Vue 3 Composition API 风格
- 组件命名使用 PascalCase（如 `ChatMessage.vue`）
- 组件文件使用 PascalCase（如 `SessionList.vue`）
- 优先使用 `<script setup>` 语法
- 本仓库未配置 ESLint/Prettier；提交前以 `npm run build` 和 Node 测试为准

### Java

- 遵循阿里巴巴 Java 开发手册
- 类命名使用 PascalCase，方法使用 camelCase
- 使用 Lombok 构造器注入（`@RequiredArgsConstructor`）与 `@Slf4j`
- **重要**：修改 Mapper 接口时，必须同步更新 XML 文件
  - XML 位置：`src/main/resources/com/rag/ragserver/mapper/`
- 本仓库未配置 Checkstyle/Spotless；提交前以 `mvn -o test` 为准

### Python

- 遵循 PEP 8 规范
- 使用 type hints
- 本仓库未配置 black/flake8；提交前以对应模块的 `pytest` 为准

## 提交规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type 类型

- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式（不影响代码运行）
- `refactor`: 重构（既不是新增功能，也不是修改bug）
- `perf`: 性能优化
- `test`: 增加测试
- `chore`: 构建过程或辅助工具的变动
- `security`: 安全相关修复

### 示例

```bash
feat(chat): add streaming response support

- Implement SSE for real-time chat streaming
- Update frontend to handle streaming data
- Add retry mechanism for failed streams

Closes #123
```

## 问题反馈

### Issue 标签

- `bug`: 程序错误
- `enhancement`: 功能增强
- `documentation`: 文档相关
- `question`: 问题咨询
- `good first issue`: 适合新手
- `help wanted`: 需要帮助

### Pull Request 检查清单

提交 PR 前请确认：

- [ ] 代码遵循项目规范
- [ ] 通过所有测试（如有）
- [ ] 添加了必要的测试（如有新功能）
- [ ] 更新了相关文档（README、SECURITY 等）
- [ ] PR 描述清晰，关联了相关 Issue
- [ ] Commit 信息符合 Conventional Commits 规范
- [ ] 没有合并冲突
- [ ] **没有提交敏感信息**（密码、API Keys 等）
- [ ] 配置文件使用了 `.example` 模板
- [ ] MyBatis XML 文件已同步更新（如修改了 Mapper）

## 📞 联系方式

如有任何问题，欢迎通过以下方式联系：

- **GitHub Issues**: [提交 Issue](https://github.com/cockmake/general-rag-system/issues)
- **GitHub Discussions**: [参与讨论](https://github.com/cockmake/general-rag-system/discussions)

## 📚 参考资料

- [Vue 3 风格指南](https://vuejs.org/style-guide/)
- [阿里巴巴 Java 开发手册](https://github.com/alibaba/p3c)
- [PEP 8 - Python 代码风格指南](https://pep8.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [语义化版本](https://semver.org/)

## 📄 许可证

通过提交代码，您同意您的贡献将使用 [Apache 2.0 许可证](LICENSE)。

---

再次感谢您的贡献！🎉
