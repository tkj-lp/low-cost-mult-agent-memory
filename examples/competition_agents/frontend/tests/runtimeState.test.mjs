import { test } from 'node:test'
import assert from 'node:assert/strict'
import { taskExecutionState, eventsForRun, comparableRuns } from '../src/runtimeState.mjs'

test('未执行及失败任务不能显示成功', () => {
  assert.equal(taskExecutionState([], { final_answer: 'stale' }), 'waiting')
  assert.equal(taskExecutionState([{event_type:'task_start'}], null, 'running'), 'running')
  assert.equal(taskExecutionState([{event_type:'task_done',return_result:{error:'bad'}}]), 'failed')
  assert.equal(taskExecutionState([{event_type:'task_failed'}]), 'failed')
  assert.equal(taskExecutionState([{event_type:'task_done'}]), 'done')
  assert.equal(taskExecutionState([{event_type:'task_start'}], null, 'stopped'), 'stopped')
  assert.equal(taskExecutionState([], null, 'stopped'), 'waiting')
})
test('当前运行过滤历史与无归属事件', () => {
  assert.deepEqual(eventsForRun([{run_id:'old'},{run_id:'new'},{}], 'new'), [{run_id:'new'}])
  assert.deepEqual(eventsForRun([{run_id:'old'}], ''), [])
})
test('仅相同模型、数据、版本和完整流程可比较耗时', () => {
  const record = { model:'same', backend:'local', datasetHash:'a', knowledgeHash:'b', baselineVersion:'v3', answerMetric:'coverage', taskCount:3, timingValid:true }
  assert.equal(comparableRuns(record, {...record,collaborationMode:'structured'}), true)
  for (const key of ['model','backend','datasetHash','knowledgeHash','baselineVersion','answerMetric','taskCount','timingValid']) {
    assert.equal(comparableRuns(record,{...record,[key]:'different'}),false)
  }
})
