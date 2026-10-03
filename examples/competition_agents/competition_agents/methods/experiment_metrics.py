"""共同指标：区分检索命中、证据供给、计算复用和驻留空间。"""


def task_metrics(result, method):
    metrics = result['metrics']
    audit = result.get('retrieval_audit') or {}
    memory = result.get('memory_read') or {}
    if isinstance(memory, dict):
        delivered = len(memory.get('evidence', {})) or int(bool(memory.get('text')))
    else:
        delivered = len(memory)
    # 证据供给仅证明旧证据进入计算，不声称已判断模型在语义上采用了它。
    metrics.setdefault('memory_evidence_delivered', delivered)
    completed = result.get('workflow_complete', not result.get('error'))
    metrics.setdefault('memory_only_tasks', int(bool(delivered) and not metrics.get('knowledge_searches') and completed))
    metrics.setdefault('memory_searches', 1)
    metrics['memory_reuse_task'] = int(bool(metrics['memory_evidence_delivered']) and completed)
    metrics['memory_correct_reuse_task'] = int(bool(metrics['memory_reuse_task']) and result.get('answer_correct', False))
    def external(message):
        return (message.get('kind') == 'external_tool' or message['from'] in ('executor', 'knowledge')
                or message['to'] in ('executor', 'knowledge'))
    messages = result.get('messages', [])
    metrics['agent_message_bytes'] = (None if method == 'latent' else
                                    sum(m['wire_bytes'] for m in messages if not external(m)))
    metrics['external_tool_message_bytes'] = sum(m['wire_bytes'] for m in messages if external(m))
    for role in ('planner', 'retriever', 'reasoner', 'summarizer', 'autoform'):
        metrics[role + '_service_ms'] = sum(c['elapsed_ms'] for c in result['calls'] if c['role'] == role)
    for key in ('generation_call_ms', 'generation_prefill_forward_ms', 'generation_decode_forward_ms',
                'appended_forward_ms', 'state_finalize_ms', 'input_processing_ms',
                'latent_steps', 'latent_forward_ms', 'intermediate_work_output_tokens',
                'tool_output_tokens', 'external_generation_calls', 'role_prefill_positions', 'role_prefill_ms'):
        metrics[key] = sum(c.get(key, 0) or 0 for c in result['calls'])
    metrics['work_reuse_decisions'] = sum(
        c.get('decision_response', c.get('response', {})).get('action') == 'reuse' for c in result['calls'])
    metrics['shared_instruction_reuses'] = sum(bool(c.get('shared_instruction_reused')) for c in result['calls'])
    metrics['communication_scope'] = 'logical_messages_or_shared_state_not_network_traffic'
    return metrics


def aggregate(rows):
    keys = ('memory_searches', 'memory_requeries', 'memory_requery_new_evidence',
            'memory_text_units_delivered', 'memory_evidence_delivered', 'memory_only_tasks', 'memory_reuse_task',
            'memory_correct_reuse_task', 'memory_evidence_used', 'facts_encoded', 'facts_reused_without_encoding',
            'forward_positions', 'active_hidden_handoff_bytes', 'external_tool_message_bytes',
            'knowledge_request_bytes', 'knowledge_result_bytes', 'lookup_query_fallbacks',
            'cache_reclaimed_positions', 'cache_reclamation_ms', 'cache_recovery_positions',
            'generation_call_ms', 'generation_prefill_forward_ms', 'generation_decode_forward_ms',
            'appended_forward_ms', 'state_finalize_ms', 'input_processing_ms',
            'latent_steps', 'latent_forward_ms', 'intermediate_work_output_tokens',
            'tool_output_tokens', 'external_generation_calls', 'role_prefill_positions', 'role_prefill_ms',
            'work_reuse_decisions', 'shared_instruction_reuses',
            'planner_service_ms', 'retriever_service_ms', 'reasoner_service_ms', 'summarizer_service_ms', 'autoform_service_ms')
    result = {key: sum(r['metrics'].get(key, 0) or 0 for r in rows) for key in keys}
    count = len(rows)
    result['memory_reuse_task_rate'] = result['memory_reuse_task']/count
    result['memory_only_task_rate'] = result['memory_only_tasks']/count
    result['full_answer_rate'] = sum(bool(r['answer_correct']) for r in rows)/count
    result['agent_message_bytes'] = (sum(r['metrics']['agent_message_bytes'] for r in rows)
                                     if all(r['metrics']['agent_message_bytes'] is not None for r in rows) else None)
    for kind in ('memory', 'knowledge'):
        calls = sum(r['metrics'].get(kind + '_searches', 0) for r in rows)
        milliseconds = sum(r['metrics'].get(kind + '_search_ms', 0) for r in rows)
        result[kind + '_search_mean_ms'] = milliseconds/calls if calls else None
    result['network_transfer_bytes'] = None
    result['memory_reuse_definition'] = 'previous_evidence_delivered_to_completed_workflow; not causal attribution'
    return result
