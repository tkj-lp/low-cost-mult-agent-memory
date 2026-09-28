# 启动与复核

以下命令在 `examples/competition_agents` 目录执行。使用已经安装本地模型和检索模型的运行环境。

## 前端

先启动后端：

```powershell
.\start_local_runner.ps1 -PythonPath 'D:\anaconda\envs\langgraph\python.exe'
```

另开终端启动前端：

```powershell
cd frontend
npm.cmd run dev -- --host 127.0.0.1
```

访问 `http://127.0.0.1:5173`。五个协作选项共享同一份配置；选择后自动固定记忆和检索。隐状态只显示已校准的本地模型。当前连续样本为用户截图中第三个 TopiOCQA 文件、第153号会话的完整10题；前端固定执行10轮。代码功能库保留20项，前端按原順序跑前10项。

## 命令行

```powershell
python run_competition.py --method plain --backend local --limit 10
python run_competition.py --method structured --backend openai-compatible --remote-model qwen3.7-flash-2026-07-15 --limit 10
python run_competition.py --method latent --backend local --limit 10
python run_competition.py --method structured --dataset codeact --backend openai-compatible --limit 10
```

另外两种对照的选项值是 `autoform`、`llmlingua`。在线地址、模型和密钥从本机 `.env.local` 读取，或由前端本次运行配置传入；示例配置只含占位值。

本地基础模型与检索模型须预先下载。LLMLingua-2 使用官方 `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`；缺少模型时明确报错。本机已安装依赖并完成官方模型单次压缩检查。

输出默认写入 `runtime/runs`，每次运行独立目录。知识库、校准和模型初始化时间记为 `setup_ms_excluded`。未完整走完的任务保留错误和已有调用，不能算作完整成功耗时。

默认跨运行不复用经验，保证新对照从独立状态开始。若明确需要长期经验，可设置 `COMPETITION_LONG_TERM_MEMORY=1` 后重启后端，数据库按方案和模型隔离；前端清空经验不会删除预备知识或历史结果。

## 固定版本与测试

```powershell
python -m tools.verify_locked_baselines
python -m pytest tests hypergraph_memory_method/tests -q
cd frontend
npm.cmd run build
```

核验同时检查源代码、角色提示词、前端配置、预备知识和校准文件。本轮统一十题入口，发布第五版；以后修改方案发布新版本时运行 `python -m tools.verify_locked_baselines --write` 更新清单。

## 离线准备

正常运行读取已发布的知识库与映射。截图数据的来源准备入口为 `python -m tools.prepare_ranked_continuous`，随后使用 `python -m tools.build_prepared_knowledge` 和 `python -m tools.prepare_latent_readout`；各工具参数见 `--help`。使用新目录，不能覆盖现有固定产物。抽取校验工具也集中在 `tools`。

以下为历史三题调试入口，不作为十题最终实验入口： `python -m tools.validate_ranked_three --output runtime/新的结果目录`。按顺序运行在线文本、在线协议、本地文本、本地协议、本地隐状态；不并发争抢推理资源，不把在线和本地时间放在同一比较组。

只修改协议时，可加 `--method structured` 仅复测相同三题的在线与本地协议；也可用 `--backend local` 或 `--backend openai-compatible` 指定调用方式。每次须使用新输出目录，保留前版失败。字段归一化时间、被拒绝的引用、依据原始字段继续交接的记录，分别保存在每题的 `protocol_parsing`、`protocol_rejections` 和指标中。

当前校准来自预备知识生成的样本，尚未证明对新模型或任意任务泛化；更换模型必须单独校准和验收。


前端状态检查：在前端目录执行 `node --test tests/runtimeState.test.mjs`。启动前后端服务后可执行 `node tests/ui-smoke.cjs` 检查失败、停止、新运行和旧结果隔离；此测试使用拦截的测试事件，不调用在线模型，也不往真实结果目录写虚构成绩。

服务器上传、预检和完整十题运行见 [服务器部署说明](docs/server_deployment.md)。命令行省略 `--limit` 时与前端一致运行十题；显式数量仅用于调试。
