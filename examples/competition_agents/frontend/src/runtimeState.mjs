// 结果存在并不等于成功；只有完成事件或明确完成的结果能结束任务。
export function taskExecutionState(events, result, runStatus = '') {
  if (result?.error || result?.status === 'failed' || events.some(e =>
    e.event_type === 'task_failed' || (e.event_type === 'task_done' && e.return_result?.error))) return 'failed'
  if (result?.status === 'completed' || events.some(e => e.event_type === 'task_done' &&
    e.status !== 'failed' && !e.return_result?.error)) return 'done'
  const started = events.some(e => e.event_type === 'task_start')
  if (started && ['stopped', 'failed', 'completed'].includes(runStatus)) return 'stopped'
  return started ? 'running' : 'waiting'
}

export function eventsForRun(events, runId) {
  return runId ? events.filter(event => event.run_id === runId) : []
}

export function comparableRuns(left, right) {
  const keys = ['model', 'backend', 'datasetHash', 'knowledgeHash', 'baselineVersion', 'answerMetric', 'taskCount']
  return Boolean(left && right && left.timingValid === true && right.timingValid === true && keys.every(key =>
    left[key] != null && left[key] !== '' && left[key] === right[key]))
}
