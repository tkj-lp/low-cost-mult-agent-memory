import { after, test } from 'node:test'
import assert from 'node:assert/strict'
import { fileURLToPath } from 'node:url'
import { createServer } from 'vite'

const server = await createServer({ root: fileURLToPath(new URL('..', import.meta.url)), configFile: false,
  optimizeDeps: { noDiscovery: true, include: [] }, server: { middlewareMode: true } })
after(() => server.close())
const options = await server.ssrLoadModule('/src/experimentOptions.js')

test('默认与旧三题配置均选择完整十题', () => {
  assert.equal(options.defaultExperimentConfig().taskCount, 10)
  for (const dataset of options.datasets) {
    const config = options.normalizeExperimentConfig({ ...options.defaultExperimentConfig(), dataset: dataset.value, taskCount: 3 })
    assert.equal(config.taskCount, 10)
    assert.match(options.buildExperimentCommand(config), /'--limit' '10'/)
  }
})

test('数据集按完整路径匹配，旧题目文件不能冒充当前十题', () => {
  const current = options.datasets[0]
  assert.equal(options.datasetKeyForFile(`E:\\project\\${current.file.replaceAll('/', '\\')}`), current.value)
  const old = 'data/continuous_topiocqa_153_v1/questions.json'
  assert.equal(options.datasetKeyForFile(old), old)
  const imported = options.configFromRunResult({ dataset_file: old, task_count: 3, experiment_mode: 'baseline_text' }, options.defaultExperimentConfig())
  assert.equal(imported.dataset, old)
  assert.equal(imported.taskCount, 3)
  assert.equal(options.normalizeExperimentConfig(imported).taskCount, 10)
})

test('三种主方案保留原有记忆搭配和后端限制', () => {
  for (const method of ['plain', 'structured', 'latent']) {
    const config = options.normalizeExperimentConfig({ ...options.defaultExperimentConfig(), collaborationMode: method, backend: 'openai-compatible' })
    assert.equal(config.taskCount, 10)
    assert.equal(config.retrievalMode, method === 'plain' ? 'semantic' : 'hypergraph')
    if (method === 'latent') assert.equal(config.backend, 'local')
  }
})
