const allDatasets = [
  {
    label: 'TopiOCQA 第 1 组（10轮）',
    value: 'topiocqa-1',
    taskCount: 10,
    datasetType: 'continuous',
    latentDatasetType: 'continuous',
    file: 'examples/competition_agents/data/topiocqa_1_10turns.json'
  },
  {
    label: 'TopiOCQA 第 2 组（10轮）',
    value: 'topiocqa-2',
    taskCount: 10,
    datasetType: 'continuous',
    latentDatasetType: 'continuous',
    file: 'examples/competition_agents/data/topiocqa_2_10turns.json'
  },
  {
    label: 'TopiOCQA \u5b98\u65b9\u7b2c 1 \u7ec4\uff0810\u8f6e\uff09',
    value: 'topiocqa-official-1',
    taskCount: 10,
    datasetType: 'continuous',
    latentDatasetType: 'continuous',
    file: 'examples/competition_agents/data/official_topiocqa_001_10turns.json'
  },
  {
    label: 'TopiOCQA \u5b98\u65b9\u7b2c 2 \u7ec4\uff0810\u8f6e\uff09',
    value: 'topiocqa-official-2',
    taskCount: 10,
    datasetType: 'continuous',
    latentDatasetType: 'continuous',
    file: 'examples/competition_agents/data/official_topiocqa_002_10turns.json'
  },
  {
    label: 'TopiOCQA \u5b98\u65b9\u7b2c 3 \u7ec4\uff0810\u8f6e\uff09',
    value: 'topiocqa-official-3',
    taskCount: 10,
    datasetType: 'continuous',
    latentDatasetType: 'continuous',
    file: 'examples/competition_agents/data/official_topiocqa_003_10turns.json'
  },
  {
    label: 'QReCC \u5b98\u65b9\u7b2c 1 \u7ec4\uff0810\u8f6e\uff09',
    value: 'qrecc-official-1',
    taskCount: 10,
    datasetType: 'continuous',
    latentDatasetType: 'continuous',
    file: 'examples/competition_agents/data/official_qrecc_001_10turns.json'
  },
  {
    label: 'QReCC \u5b98\u65b9\u7b2c 2 \u7ec4\uff0810\u8f6e\uff09',
    value: 'qrecc-official-2',
    taskCount: 10,
    datasetType: 'continuous',
    latentDatasetType: 'continuous',
    file: 'examples/competition_agents/data/official_qrecc_002_10turns.json'
  },
  {
    label: 'QReCC \u5b98\u65b9\u7b2c 3 \u7ec4\uff0810\u8f6e\uff09',
    value: 'qrecc-official-3',
    taskCount: 10,
    datasetType: 'continuous',
    latentDatasetType: 'continuous',
    file: 'examples/competition_agents/data/official_qrecc_003_10turns.json'
  },
  {
    label: '\u5b98\u65b9 TopiOCQA + QReCC\uff086\u7ec4 / 60\u8f6e\uff09',
    value: 'official-topiocqa-qrecc-6chains',
    taskCount: 60,
    datasetType: 'continuous',
    latentDatasetType: 'continuous',
    file: 'examples/competition_agents/data/official_topiocqa_qrecc_6chains_10turns.json'
  },
  ...[1, 2, 3].map((index) => ({
    label: `TopiOCQA \u5b98\u65b9\u4e8b\u5b9e\u578b\u7b5b\u9009\u7b2c ${index} \u7ec4\uff0810\u8f6e\uff09`,
    value: `topiocqa-factoid-ranked-${index}`,
    taskCount: 10,
    datasetType: 'continuous',
    latentDatasetType: 'continuous',
    file: `examples/competition_agents/data/official_factoid_ranked_topiocqa_${String(index).padStart(3, '0')}_10turns.json`
  })),
  ...[1, 2, 3].map((index) => ({
    label: `QReCC \u5b98\u65b9\u4e8b\u5b9e\u578b\u7b5b\u9009\u7b2c ${index} \u7ec4\uff0810\u8f6e\uff09`,
    value: `qrecc-factoid-ranked-${index}`,
    taskCount: 10,
    datasetType: 'continuous',
    latentDatasetType: 'continuous',
    file: `examples/competition_agents/data/official_factoid_ranked_qrecc_${String(index).padStart(3, '0')}_10turns.json`
  })),
  {
    label: 'CodeAct 第 1 组（10题）',
    value: 'codeact-1',
    taskCount: 10,
    datasetType: 'codeact',
    latentDatasetType: 'codeact',
    file: 'examples/competition_agents/data/codeact_1_10tasks.json'
  },
  {
    label: 'CodeAct 第 2 组（10题）',
    value: 'codeact-2',
    taskCount: 10,
    datasetType: 'codeact',
    latentDatasetType: 'codeact',
    file: 'examples/competition_agents/data/codeact_2_10tasks.json'
  }
]

export const datasets = allDatasets.filter((item) =>
  /^(topiocqa-[12]|topiocqa-factoid-ranked-[123]|qrecc-factoid-ranked-[123]|codeact-[12])$/.test(item.value)
)

export const backends = [
  { label: '本地模型', value: 'local' },
  { label: 'OpenAI 兼容接口', value: 'openai-compatible' }
]

export const localModels = [
  { label: 'Qwen2.5 0.5B Instruct', value: 'Qwen/Qwen2.5-0.5B-Instruct' },
  { label: 'Qwen2.5 1.5B Instruct', value: 'Qwen/Qwen2.5-1.5B-Instruct' },
  { label: 'TinyLlama 1.1B Chat', value: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' },
  { label: 'SmolLM2 1.7B Instruct', value: 'HuggingFaceTB/SmolLM2-1.7B-Instruct' },
  { label: 'Phi-3.5 Mini Instruct', value: 'microsoft/Phi-3.5-mini-instruct' },
  { label: 'DeepSeek Coder 1.3B Instruct', value: 'deepseek-ai/deepseek-coder-1.3b-instruct' }
]

export const collaborationModes = [
  { label: '纯文本', value: 'plain', icon: '⊙', experimentMode: 'baseline_text' },
  { label: 'AutoForm', value: 'autoform', icon: 'A', experimentMode: 'autoform_text' },
  { label: 'LLMLingua-2', value: 'llmlingua', icon: 'L', experimentMode: 'llmlingua_text' },
  { label: '结构化协议', value: 'structured', icon: '▣', experimentMode: 'protocol_api' },
  { label: '协议 + 向量', value: 'protocol-vector', icon: '⌘', experimentMode: 'protocol_api_embedding' },
  { label: '隐状态传递', value: 'latent', icon: '◷', experimentMode: 'hidden_api' }
]

export const memoryModes = [
  { label: '普通文本记忆', value: 'text', icon: '⊙' },
  { label: '超图记忆', value: 'hypergraph', icon: '⌘' }
]

export const textRetrievalModes = [
  { label: '关键词', value: 'keyword', icon: '⌕' },
  { label: '标签', value: 'tag', icon: '◇' },
  { label: '语义相似度', value: 'semantic', icon: '⌁' }
]

export const hypergraphRetrievalModes = [
  { label: '超图跳跃', value: 'hypergraph', icon: '⌘' },
  { label: '融合检索', value: 'fusion', icon: '▰' }
]

export const agentSequence = ['规划智能体', '检索智能体', '推理智能体', '总结智能体']

export function defaultExperimentConfig() {
  return {
    dataset: 'topiocqa-factoid-ranked-1',
    taskCount: 10,
    backend: 'local',
    localModel: 'Qwen/Qwen2.5-0.5B-Instruct',
    baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    remoteModel: '',
    apiKey: '',
    collaborationMode: 'plain',
    memoryMode: 'text',
    retrievalMode: 'keyword'
  }
}

export function labelOf(options, value, fallback = value) {
  return options.find((item) => item.value === value)?.label ?? fallback
}

export function selectedDataset(config) {
  return datasets.find((item) => item.value === config.dataset) ?? datasets[0]
}

export function datasetKeyForFile(file) {
  const text = String(file || '')
  if (allDatasets.some((item) => item.value === text)) return text
  const matched = allDatasets.find((item) => text.includes(item.file.split('/').pop()))
  return matched?.value ?? text
}

export function selectedCollaboration(config) {
  return collaborationModes.find((item) => item.value === config.collaborationMode) ?? collaborationModes[0]
}

export function retrievalOptionsFor(memoryMode) {
  return memoryMode === 'hypergraph' ? hypergraphRetrievalModes : textRetrievalModes
}

export function defaultRetrievalFor(memoryMode) {
  return retrievalOptionsFor(memoryMode)[0].value
}

export function normalizeExperimentConfig(config) {
  const dataset = selectedDataset(config)
  const retrievalOptions = retrievalOptionsFor(config.memoryMode)
  const retrievalMode = retrievalOptions.some((item) => item.value === config.retrievalMode)
    ? config.retrievalMode
    : retrievalOptions[0].value

  return {
    ...config,
    dataset: dataset.value,
    taskCount: Math.max(1, Number(config.taskCount) || dataset.taskCount),
    retrievalMode
  }
}

export function buildMatrixRows(config) {
  const normalized = normalizeExperimentConfig(config)
  const dataset = selectedDataset(normalized)
  const collaboration = selectedCollaboration(normalized)
  const memoryLabel = labelOf(memoryModes, normalized.memoryMode)

  return retrievalOptionsFor(normalized.memoryMode).map((retrieval) => ({
    id: [
      normalized.dataset,
      normalized.collaborationMode,
      normalized.memoryMode,
      retrieval.value
    ].join('|'),
    dataset: dataset.label,
    taskCount: normalized.taskCount,
    collaborationMode: collaboration.label,
    memoryMode: memoryLabel,
    retrievalMode: retrieval.label,
    retrievalValue: retrieval.value,
    status: retrieval.value === normalized.retrievalMode ? '当前配置' : '待运行',
    charCost: '--',
    duration: '--',
    reuseRate: '--',
    isSelected: retrieval.value === normalized.retrievalMode,
    config: {
      ...normalized,
      retrievalMode: retrieval.value
    }
  }))
}

export function buildExperimentCommand(config) {
  const normalized = normalizeExperimentConfig(config)
  const dataset = selectedDataset(normalized)
  const collaboration = selectedCollaboration(normalized)
  const script = 'examples/competition_agents/run_demo.py'
  const contextSource = dataset.datasetType === 'codeact' ? 'inline' : 'hypergraph'
  const args = [
    `--experiment-mode ${collaboration.experimentMode}`,
    `--llm-backend ${normalized.backend === 'local' ? 'local' : 'api'}`,
    `--dataset-type ${dataset.datasetType}`,
    `--dataset-file ${dataset.file}`,
    `--limit 0`,
    `--context-source ${contextSource}`,
    `--memory-retrieval-mode ${normalized.retrievalMode}`,
    '--show-memory',
    '--show-latent',
    '--show-state'
  ]

  const envLines =
    normalized.backend === 'local'
      ? [`$env:LOCAL_HIDDEN_MODEL="${normalized.localModel}"`]
      : [
          normalized.baseUrl ? `$env:LLM_BASE_URL="${normalized.baseUrl}"` : '',
          normalized.remoteModel ? `$env:LLM_MODEL="${normalized.remoteModel}"` : '',
          normalized.apiKey ? '$env:LLM_API_KEY="<使用页面输入的 api_key>"' : '# 未填写 api_key 时使用 examples/competition_agents/.env.local'
        ].filter(Boolean)

  return [...envLines, `python ${script} ${args.join(' ')}`].join('\n')
}

export function parseRunResultText(raw) {
  const marker = '=== run_result_json ==='
  const idx = raw.lastIndexOf(marker)
  if (idx < 0) {
    throw new Error('未找到 run_result_json 区块')
  }
  return JSON.parse(raw.slice(idx + marker.length).trim())
}

export function decodeRunResultBuffer(buffer) {
  const bytes = new Uint8Array(buffer)
  if (bytes.length >= 2 && bytes[0] === 0xff && bytes[1] === 0xfe) {
    return new TextDecoder('utf-16le').decode(bytes)
  }
  if (bytes.length >= 2 && bytes[0] === 0xfe && bytes[1] === 0xff) {
    return new TextDecoder('utf-16be').decode(bytes)
  }
  return new TextDecoder('utf-8').decode(bytes)
}

export function configFromRunResult(payload, fallbackConfig) {
  const modeToCollaboration = {
    baseline_text: 'plain',
    autogen_text: 'autogen',
    camel_text: 'camel',
    autogen_camel_text: 'autogen',
    autoform_text: 'autoform',
    llmlingua_text: 'llmlingua',
    protocol_api: 'structured',
    protocol_api_embedding: 'protocol-vector',
    hidden_api: 'latent',
    latent_hidden: 'latent'
  }
  const firstRun = Array.isArray(payload.runs) ? payload.runs[0] ?? {} : {}
  const firstTask = Array.isArray(firstRun.task_results) ? firstRun.task_results[0] ?? {} : {}
  const inferredRetrievalMode =
    payload.memory_retrieval_mode ||
    firstRun.memory_retrieval_mode ||
    firstTask.memory_retrieval_mode ||
    (payload.context_source === 'hypergraph' ? 'hypergraph' : fallbackConfig.retrievalMode)
  const dataset = datasets.find((item) => payload.dataset_file?.endsWith(item.file.split('/').pop()))
  const memoryMode = ['hypergraph', 'fusion'].includes(inferredRetrievalMode) ? 'hypergraph' : 'text'

  return normalizeExperimentConfig({
    ...fallbackConfig,
    dataset: dataset?.value ?? fallbackConfig.dataset,
    taskCount: payload.task_count || fallbackConfig.taskCount,
    collaborationMode: modeToCollaboration[payload.experiment_mode] ?? fallbackConfig.collaborationMode,
    memoryMode,
    retrievalMode: inferredRetrievalMode
  })
}
