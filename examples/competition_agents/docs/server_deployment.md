# 服务器部署与十题实验

正式配置为 `competition-locked-v5-10turns`。连续问答采用用户指定分支的第153号会话，原顺序全部10题。代码执行入口运行原库前10项，原20项全部保留。两类任务分别统计，不混合比较。前端、运行服务和命令行默认数量一致；命令行显式指定 `--limit` 只用于调试。

十题公开问题位于 `data/continuous_topiocqa_153_10turns_v1/questions.json`，标准答案单独存放。知识库仍为 `topiocqa_153_v2` 的7个完整章节、252条事实，隐状态映射仍为 `topiocqa_153_readout_v2`。没有重新抽取或修改三种协作机制。旧三题数据和已有实验记录保留，不能当作新版十题成绩。

## 上传范围

发布包包含整个仓库的源代码、库、前端源码及构建页面、数据、预备知识、校准文件、测试、文档和保留的实验材料。仅排除版本库内部文件、已安装依赖、环境密钥、模型下载缓存和临时运行产物。每个文件的校验值记录在包内 `SERVER_RELEASE.json`。本地旧结果仍在原处，必要的历史证据保留于 `docs/evidence`。

上传到 `/root/autodl-tmp/project/competition-20260928-v5`，不覆盖原有 `low-cost-mult-agent-memory` 目录。基础模型权重不包含在源码包中。

## 环境与预检

以下命令在新项目的 `examples/competition_agents` 目录运行。使用独立环境，不直接升级服务器已有工程的依赖。根据实际显卡驱动安装兼容的显卡版张量计算库，再安装 `requirements.txt`。本机依赖版本和服务器实际安装版本应随正式结果一起保留，不能混用两台机器的耗时比较。

```bash
python -m pip install -r requirements.txt
export HF_HOME=/root/autodl-tmp/huggingface
python -m tools.server_preflight --files-only
python -m tools.server_preflight --require-gpu
```

预先下载所需模型：

```bash
hf download Qwen/Qwen2.5-1.5B-Instruct
hf download sentence-transformers/all-MiniLM-L6-v2
hf download microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
```

当前隐状态映射只适配上述千问模型，不能直接换成服务器已有的其他模型。预检通过只能说明文件、依赖、缓存和设备已准备好，正式准确率与耗时仍需真实实验验证。

## 正式入口

三种方案分别启动独立进程，避免同时争抢显卡：

```bash
python run_competition.py --method plain --backend local
python run_competition.py --method structured --backend local
python run_competition.py --method latent --backend local
```

以上均默认完整10题。结果写入各自的 `runtime/runs` 新目录；不完整流程不会伪装成完整成功。代码任务使用 `--dataset codeact`，当前仅支持纯文本和协议字段。其他两种对照仍保留，但不自动混入这三条基线。

## 远程页面

服务器运行后端和已构建的前端页面，无须为了浏览重新安装前端依赖：

```bash
python runner_server.py
# 另开终端，从同一比赛目录执行：
python -m http.server 5173 --bind 127.0.0.1 --directory frontend/dist
```

本机通过隧道访问：

```powershell
ssh -p 45468 -L 15173:127.0.0.1:5173 -L 8765:127.0.0.1:8765 root@connect.cqa1.seetacloud.com
```

浏览器打开 `http://127.0.0.1:15173/`。本机原运行服务使用8765端口时，需要先停止它，才能把该端口转发到服务器。运行服务只绑定本机地址，不需要暴露到公网。页面实际连接哪个后端应在最终实验前核对，不能将本机结果误认成服务器结果。

首次接入时，服务器未挂载显卡设备，张量计算库报告可用显卡数量为零；上传和文件验证可完成，显卡实验须等待实例启用显卡。
