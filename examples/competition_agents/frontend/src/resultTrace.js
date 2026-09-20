const traces = new Map()

export async function loadResultTrace(record) {
  const key = `${record.source}|${record.runIndex ?? 0}|${record.fingerprint || record.recordId}`
  if (record.payload?.runs?.[0]?.task_results?.length) return record.payload
  if (!traces.has(key)) {
    const request = fetch('http://127.0.0.1:8765/api/runner/trace', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: record.source, runIndex: record.runIndex ?? 0 })
    }).then(async (response) => {
      if (!response.ok) throw new Error('Trace request failed')
      const { payload } = await response.json()
      if (!payload?.runs?.[0]?.task_results?.length) throw new Error('Missing task details')
      return payload
    }).catch((error) => {
      traces.delete(key)
      throw error
    })
    traces.set(key, request)
    if (traces.size > 3) traces.delete(traces.keys().next().value)
  }
  return traces.get(key)
}
