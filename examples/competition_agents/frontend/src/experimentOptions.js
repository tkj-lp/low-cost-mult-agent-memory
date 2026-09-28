import catalog from '../../config/methods.json'

const allDatasets = catalog.datasets
export const datasets = allDatasets

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

export const collaborationModes = catalog.methods.map((item) => ({ ...item, icon: '●' }))
export const memoryModes = [
  { label: '普通文本记忆', value: 'text', icon: '⊙' },
  { label: '超图记忆', value: 'hypergraph', icon: '⌘' }
]
export const textRetrievalModes = catalog.retrievalModes.filter((item) => item.value === 'semantic')
export const hypergraphRetrievalModes = catalog.retrievalModes.filter((item) => item.value === 'hypergraph')

export const agentSequence = ['规划智能体', '检索智能体', '推理智能体', '总结智能体']

export function defaultExperimentConfig() {
  return {
    dataset: catalog.defaultDataset,
    taskCount: catalog.defaultTaskCount,
    backend: 'local',
    localModel: catalog.defaultLocalModel,
    baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    remoteModel: catalog.defaultRemoteModel,
    apiKey: '',
    collaborationMode: 'plain',
    memoryMode: 'text',
    retrievalMode: 'semantic'
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
  const matched = allDatasets.find((item) => text.replaceAll('\\', '/').includes(item.file))
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
  const requested = selectedCollaboration(config)
  const method = dataset.methods && !dataset.methods.includes(requested.value) ? collaborationModes.find(m => m.value === dataset.methods[0]) : requested
  return {
    ...config,
    dataset: dataset.value,
    collaborationMode: method.value,
    taskCount: dataset.runTaskCount ?? dataset.taskCount,
    memoryMode: method.memory,
    retrievalMode: method.retrieval,
    backend: method.backends.includes(config.backend) ? config.backend : method.backends[0],
    localModel: method.models && !method.models.includes(config.localModel) ? method.models[0] : config.localModel
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
  const value = normalizeExperimentConfig(config)
  const quote = (text) => "'" + String(text).replaceAll("'", "''") + "'"
  const args = [
    '--method', value.collaborationMode, '--dataset', value.dataset,
    '--backend', value.backend, '--local-model', value.localModel,
    '--remote-model', value.remoteModel, '--limit', String(value.taskCount)
  ].map(quote)
  return 'python examples/competition_agents/run_competition.py ' + args.join(' ')
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
  const dataset = datasets.find((item) => item.value === datasetKeyForFile(payload.dataset_file))
  const memoryMode = ['hypergraph', 'fusion'].includes(inferredRetrievalMode) ? 'hypergraph' : 'text'

  const normalized = normalizeExperimentConfig({
    ...fallbackConfig,
    dataset: dataset?.value ?? fallbackConfig.dataset,
    taskCount: payload.task_count || fallbackConfig.taskCount,
    collaborationMode: modeToCollaboration[payload.experiment_mode] ?? fallbackConfig.collaborationMode,
    memoryMode,
    retrievalMode: inferredRetrievalMode
  })
  // 历史结果仍保留原始路径和实际题数；下次启动时才转为当前十题配置。
  return {
    ...normalized,
    dataset: dataset?.value ?? (payload.dataset_file || normalized.dataset),
    taskCount: payload.task_count ?? firstRun.task_count ?? normalized.taskCount
  }
}
