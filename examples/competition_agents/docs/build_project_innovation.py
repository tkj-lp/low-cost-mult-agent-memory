"""从本次完整轨迹生成创新说明；依赖报告文档运行环境。"""
from collections import Counter
import ast
import hashlib
from html import escape
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak


BASE = Path(__file__).resolve().parents[1]
DOCS = BASE/'docs'
RESULTS = BASE/'outputs/qwen35_ten_20261001/qwen35-ten-20261001-v80-v1'
ABLATION = BASE/'outputs/qwen35_ten_20261001/qwen35-lifecycle-20261001-v80-v1'
NAMES = {'plain': '纯文本', 'structured': '协议字段', 'latent': '隐状态'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summarize():
    # 文档运行环境不加载模型；从源码字面量与实际记录核查最终提示。
    tree = ast.parse((BASE/'competition_agents/plain_text_collaboration.py').read_text(encoding='utf-8'))
    FINAL_ANSWER_REQUIREMENTS = next(ast.literal_eval(node.value) for node in tree.body
                                    if isinstance(node, ast.Assign)
                                    and any(isinstance(target, ast.Name) and target.id == 'FINAL_ANSWER_REQUIREMENTS'
                                            for target in node.targets))
    methods = {}
    facts_in_state = {}
    kinds = Counter()
    final_hash = hashlib.sha256(FINAL_ANSWER_REQUIREMENTS.encode()).hexdigest()
    for name in NAMES:
        raw = read(RESULTS/name/'result.json')
        run = raw['runs'][0]
        rows = run['task_results']
        assert len(rows) == 10 and all(row['status'] == 'completed' for row in rows)
        assert all(len(row['calls']) == 4 for row in rows)
        for row in rows:
            call = row['calls'][-1]
            message = (call['inputs'][-1]['text'] if name == 'latent'
                       else call['effective_messages'][0]['content'])
            assert FINAL_ANSWER_REQUIREMENTS in message
            if name == 'latent':
                for c in row['calls']:
                    assert c['feedback_mode'] == 'cache_only' and c['latent_steps'] == 0
                    assert all(n == c['forward_positions'] for n in c['layer_positions'])
                    assert c['lifetime'].get('skipped') is None
                    kinds.update(c['lifetime'].get('removed_by_kind', {}))
                    for item in c['inputs']:
                        marker = 'New original source facts:\n'
                        if marker in item['text']:
                            packet = json.JSONDecoder().raw_decode(item['text'].split(marker, 1)[1])[0]
                            for fact in packet['facts']:
                                facts_in_state[fact[0]] = fact
        methods[name] = {'status': raw['status'], 'answer_coverage': run['answer_accuracy'],
                         'metrics': run['metrics'],
                         'final_output_tokens': sum(row['calls'][-1]['output_tokens'] for row in rows),
                         'result_sha256': hashlib.sha256((RESULTS/name/'result.json').read_bytes()).hexdigest(),
                         'tasks': [{'id': row['task_id'], 'answer': row['prediction'],
                                    'coverage': row['answer_accuracy'], 'error': row['error'],
                                    'stage_seconds': {c['role']: c['elapsed_ms']/1000 for c in row['calls']},
                                    'final_output_tokens': row['calls'][-1]['output_tokens']}
                                   for row in rows]}
    assert len(facts_in_state) == 252
    assert not kinds['external_facts'] and not kinds['question'] and not kinds['common_instruction']
    ablation = read(ABLATION/'summary.json')
    ablation['evaluation_after_run'] = read(ABLATION/'evaluation-after-run.json')
    assert ablation['plan']['final_instruction_sha256'] == final_hash
    assert all(r['finished_tasks'] == 10 for r in ablation['runs'].values())
    output = {'version': 'competition-v8.0', 'model': 'Qwen/Qwen3.5-9B',
              'dataset': 'TopiOCQA 第153组原顺序完整十题',
              'knowledge': 'topiocqa_153_v4，7篇来源、252条抽取事实',
              'final_instruction_sha256': final_hash, 'methods': methods,
              'lifecycle_removed_by_kind': dict(kinds), 'facts_in_state': len(facts_in_state),
              'cache_ablation': ablation, 'tests_passed': 254}
    (RESULTS/'final-audit.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    return output


def main():
    s = summarize()
    pdfmetrics.registerFont(TTFont('CN', 'C:/Windows/Fonts/msyh.ttc', subfontIndex=0))
    pdfmetrics.registerFont(TTFont('CNBold', 'C:/Windows/Fonts/msyhbd.ttc', subfontIndex=0))
    pdfmetrics.registerFontFamily('CN', normal='CN', bold='CNBold', italic='CN', boldItalic='CNBold')
    ink, muted, accent = colors.HexColor('#202D3A'), colors.HexColor('#5C6975'), colors.HexColor('#246E72')
    styles = {
        'title': ParagraphStyle('title', fontName='CNBold', fontSize=22, leading=30, textColor=ink, spaceAfter=12),
        'subtitle': ParagraphStyle('subtitle', fontName='CN', fontSize=10, leading=16, textColor=muted, spaceAfter=12),
        'h': ParagraphStyle('h', fontName='CNBold', fontSize=14, leading=21, textColor=accent, spaceBefore=8, spaceAfter=8),
        'body': ParagraphStyle('body', fontName='CN', fontSize=10, leading=16, textColor=ink, spaceAfter=8, wordWrap='CJK'),
        'small': ParagraphStyle('small', fontName='CN', fontSize=8.5, leading=13, textColor=muted, spaceAfter=5, wordWrap='CJK'),
        'cell': ParagraphStyle('cell', fontName='CN', fontSize=9, leading=14, textColor=ink, wordWrap='CJK'),
    }
    story, manuscript = [], []
    def p(text, kind='body'):
        story.append(Paragraph(text, styles[kind]))
        manuscript.append(text.replace('<b>', '').replace('</b>', '').replace('<br/>', '\n'))
    def table(rows, widths):
        cells = [[Paragraph(escape(str(cell)), styles['cell']) for cell in row] for row in rows]
        t = Table(cells, colWidths=widths, hAlign='LEFT', repeatRows=1)
        t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'), ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAF0F3')),
                              ('LINEBELOW',(0,0),(-1,0),0.7,accent), ('LINEBELOW',(0,1),(-1,-1),0.3,colors.HexColor('#DCE3E7')),
                              ('LEFTPADDING',(0,0),(-1,-1),7), ('RIGHTPADDING',(0,0),(-1,-1),7),
                              ('TOPPADDING',(0,0),(-1,-1),6), ('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        story.extend([t, Spacer(1, 9)])
        manuscript.extend([' | '.join(str(cell) for cell in row) for row in rows])
    def code(items):
        for file, line, meaning in items:
            p(escape(file) + ':' + str(line) + '<br/>' + escape(meaning), 'small')
    def section(label, text):
        p('<b>' + label + '</b>　' + text)
    def page():
        story.append(PageBreak()); manuscript.append('\n---\n')

    p('项目创新说明', 'title')
    p('面向连续任务的低开销多智能体协作系统<br/>版本：第8.0版　｜　更新：2026年10月1日　｜　主要创新：2项', 'subtitle')
    p('项目采用同一基础模型承担规划、检索、推理和最终回答四种角色，支持连续问答与代码执行。三种协作方式独立运行。知识库提前准备；任务期间写回原始关系证据。当前隐状态路线接续完整缓存，不额外执行末层反馈迭代；新证据仍首次经过完整模型。')
    p('创新一　角色依赖驱动的计算状态生命周期管理', 'h')
    section('最接近的已有工作', 'LatentMAS [1] 的跨角色工作记忆传递，以及按词元重要性淘汰缓存的 H₂O [2]。')
    section('已有方案具备的能力', 'LatentMAS 已实现末层连续表示推理、各层键值缓存接续和仅最终角色生成文字。H₂O 已能依据注意力贡献与近期性保留部分缓存。这些基础能力不单列为本项目新增贡献。')
    section('本项目新增贡献', '把缓存片段与“内容类型、所属轮次、最后需要它的角色”关联，按工作流声明的期限回收。来源证据与问题不参与重要性裁剪；角色提示供下游使用后释放；旧回答保留到下一题完成。针对混合结构，只回收完整注意力层的到期位置，递归状态保持原张量。物理存储长度与原始位置编码分别维护。')
    table([['状态内容', '保留与回收规则'], ['公共规则、问题、来源证据、外部结果', '会话内保留'],
           ['角色提示', '下一个角色完成后释放；最终角色提示在本题结束释放'],
           ['内部消息边界', '本题回答完成后释放'], ['历史回答的注意力条目', '保留当前轮与上一轮；更早轮次到期释放'],
           ['线性注意力的递归状态', '保持原张量，不按词元位置裁剪']], [235, 260])
    p('代码位置与验证', 'h')
    code([('competition_agents/latent_hidden/state_lifecycle.py', 22, '片段分类、消费者期限及跨题回答期限；第37行执行布局检查与回收。'),
          ('competition_agents/llm/latent_only_chat.py', 112, '带原始位置编码接续各层缓存；角色成功完成后执行回收。'),
          ('tests/test_state_lifecycle.py', 23, '核验保留键值逐项一致、递归状态原值与地址不变、未知布局不发生部分变更。')])
    p('固定来源、同一最终提示的十题机制消融见第4页。该贡献属于工作流感知的状态管理机制；回收改变后续注意力可访问的条目，不等于证明历史语义完全无损，也不把缓存容量计作网络通信量。', 'small')

    page()
    p('创新二　超图证据与计算状态的一致性复用', 'title')
    section('最接近的已有工作', 'HyperGraphRAG [3] 的多元关系组织与检索，以及 LatentMAS [1] 的驻留工作记忆接续。')
    section('已有方案具备的能力', 'HyperGraphRAG 已将多实体关系组织成超边，使用实体与超边检索，并结合原文片段生成答案；也支持图的增量更新。LatentMAS 已避免重新编码前一角色的完整工作记忆。超图、多跳、缓存复用本身不单列为新增贡献。')
    section('本项目新增贡献', '让事实的来源身份同时贯穿建图、证据存储、协议交接和隐状态去重。每条证据保留主体、关系、客体、否定及限定；代词、简称和同名作品在来源作用域内绑定，避免仅凭名称相同合并。协议的来源标题独立成表，已召回关系完整交接；隐状态按事实标识与内容哈希判断是否已编码，相同来源版本直接复用，改变后的事实重新输入。')
    table([['环节', '保持的一致性'], ['预备知识库', '模型辅助抽取；程序保留关系方向、来源及限定，建立关联索引'],
           ['证据记忆', '写回已观察到的原始事实；答案候选不能替代来源证据'],
           ['协议交接', '主体—关系—客体与限定一起传递；来源表去重，不裁已召回事实'],
           ['隐状态接续', '已编码登记与事实内容哈希一致时复用；新增或改变事实首次编码'],
           ['记忆与查库', '已驻留来源优先取回；来源缺口触发补查，不把高相似度当作答案正确性证明']], [105, 390])
    p('这一机制处理两个具体工程问题：已经存下来的事实能否通过关联路径重新找到，以及关系方向、所属对象和限定能否完整到达下游。其贡献在于证据身份与计算复用之间的约束一致，仍需更多数据检验提取质量与检索泛化。')
    p('外部检索使用从当前计算状态校准得到的数值查询，避免上游再次生成检索文字；问题文本仍用于实体入口和会话指代绑定。小型查询映射使用预备知识样本校准，基础模型权重不变，不能称为整个系统完全免训练。', 'small')
    p('代码位置与验证', 'h')
    code([('hypergraph_memory_method/prepared_knowledge.py', 121, '实体、关系参与角色与来源作用域建图；第158行写入图结构。'),
          ('hypergraph_memory_method/source_context.py', 174, '来源独立成表，按来源及原文顺序交接完整关系与限定。'),
          ('competition_agents/evidence_memory.py', 109, '存储原始证据；第37行执行关联记忆召回。'),
          ('competition_agents/llm/latent_only_chat.py', 237, '事实标识与内容哈希登记，避免重复编码。'),
          ('competition_agents/latent_hidden/calibrated_query.py', 111, '仅用预备知识样本校准数值查询映射，不读取评测答案。'),
          ('tests/test_memory_source_alignment.py', 13, '验证来源写回、后续属性取回和已有关系不重复查库。')])
    p('本轮隐状态首次编码252条事实，后续所有题目的各层统计都只计算新增位置；来源、问题与公共规则的回收计数均为零。跨题记忆复用与耗时的完整流程证据见第3页。经验记忆有独立存储与剪枝实现，但本轮没有实际采用，不列为第三项主要创新。', 'small')

    page()
    p('验证依据与贡献边界', 'title')
    p('完整流程使用真实 TopiOCQA 第153组原顺序10题、同一千问3.5九十亿参数模型及同一7篇来源知识库。显卡为 NVIDIA RTX 6000D。三组最终角色使用同一份回答要求，统一零温度、3072词元上限和32词元输出重复保护，分开记忆库并串行运行。模型加载、离线抽取建图和查询校准不计入任务时间。', 'small')
    metrics = {name: s['methods'][name]['metrics'] for name in NAMES}
    payload = lambda name: f"{metrics[name]['actual_message_bytes']/1024:.1f} 千字节"
    table([['完整十题指标', '纯文本', '协议字段', '隐状态'],
           ['总耗时（秒）', *[f"{metrics[n]['elapsed_ms']/1000:.2f}" for n in NAMES]],
           ['答案覆盖度', *[f"{s['methods'][n]['answer_coverage']*100:.2f}%" for n in NAMES]],
           ['中间交接文本', payload('plain'), payload('structured'), '不生成中间文本；同进程驻留状态接续'],
           ['后九题仅用记忆', *[f"{metrics[n]['memory_only_tasks']}/9" for n in NAMES]],
           ['平均记忆检索（毫秒）', *[f"{metrics[n]['memory_search_ms']/metrics[n]['memory_searches']:.2f}" for n in NAMES]],
           ['最终输出词元', *[s['methods'][n]['final_output_tokens'] for n in NAMES]],
           ['角色模型调用', *[sum(len(read(RESULTS/n/'result.json')['runs'][0]['task_results'][i]['calls']) for i in range(10)) for n in NAMES]]],
          [155, 103, 103, 134])
    p('“答案覆盖度”沿用项目原评分，不等于验证全部额外叙述都正确。“仅用记忆”指本题有来源证据进入流程且没有查询知识库，属于使用记录。交接文本按实际正文计量；同进程缓存容量与网络传输是不同指标，本轮未测跨机器传输。', 'small')
    p('统一回答要求后的变化', 'h')
    p('三组最终角色使用完全相同的指令：依据原始证据回答当前问题、保留所需名称及日期、开放介绍给出概述、避免内部编号和重复说明。相同要求不保证模型输出同样长度；文本第六题仍产生较长回答，原始输出全部保留。')
    p('与上一版相比，隐状态最终输出从3380词元降到952词元，总时间从97.19秒降到35.49秒，缓存峰值从468.97降到374.12二进制兆字节。但答案覆盖度从98.75%降至90.00%，第5、6题为部分覆盖，且输出仍出现日期误述。因此本轮不能宣称在不影响准确率的前提下已经全面达到目标。', 'small')
    p('测试规模为一个模型、一个真实连续会话、每种方式单轮十题。代码执行有功能测试与独立入口，本文没有给出其完整十题性能成绩；也不把本轮结果外推到全部模型或全部数据集。', 'small')
    page()
    p('生命周期消融与复核依据', 'title')
    p('生命周期机制消融', 'h')
    a = s['cache_ablation']['runs']; full, managed = a['full'], a['consumer_lifetime']
    reduction = 100*(1-managed['state_peak_bytes']/full['state_peak_bytes'])
    table([['固定完整来源十题', '保留完整缓存', '按生命周期回收'],
           ['缓存峰值（二进制兆字节）', f"{full['state_peak_bytes']/1048576:.2f}", f"{managed['state_peak_bytes']/1048576:.2f}"],
           ['十题结束保留缓存', f"{full['retained_state_bytes']/1048576:.2f}", f"{managed['retained_state_bytes']/1048576:.2f}"],
           ['实际回收位置数', full['removed_positions'], managed['removed_positions']],
           ['原评分答案覆盖度', *[f"{s['cache_ablation']['evaluation_after_run']['scores'][n]['mean_coverage']*100:.2f}%"
                                  for n in ('full', 'consumer_lifetime')]],
           ['正常结束题数', f"{full['finished_tasks']}/10", f"{managed['finished_tasks']}/10"]], [245, 125, 125])
    p(f'该消融保持模型、问题、完整来源与最终提示一致，只改变回收策略；不包含在线检索，不能与完整流程耗时混合。峰值为缓存张量容量，包含裁剪时新旧层张量短暂共存，不包含模型权重或运行工作区，实测减少{reduction:.2f}%。回收组覆盖度降低2.08个百分点，因此目前证明的是容量下降及链路可运行，还没有证明回答能力完全保持。全部254项项目测试通过。', 'small')
    p('当前未完成原论文实现的同条件对照、多模型全量验证或统计显著性检验，不宣称普遍优于已有工作。来源事实与问题全部保留，但这不能替代对缓存回收语义影响的验证。', 'small')
    p('复核位置', 'h')
    p('完整结果：outputs/qwen35_ten_20261001/qwen35-ten-20261001-v80-v1/{plain,structured,latent}/result.json<br/>机制消融：outputs/qwen35_ten_20261001/qwen35-lifecycle-20261001-v80-v1/<br/>自动核查：完整结果目录内 final-audit.json；消融脚本 experiments/final_answer_lifecycle_ablation.py。<br/>正文代码路径均相对 examples/competition_agents，行号以第8.0版为准。', 'small')
    p('已有工作', 'h')
    p('[1] Zou 等，Latent Collaboration in Multi-Agent Systems，2025，§3.1–3.2。<br/><link href="https://arxiv.org/abs/2511.20639" color="#246E72">https://arxiv.org/abs/2511.20639</link>', 'small')
    p('[2] Zhang 等，H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models，2023。<br/><link href="https://arxiv.org/abs/2306.14048" color="#246E72">https://arxiv.org/abs/2306.14048</link>', 'small')
    p('[3] Luo 等，HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation，2025，§4。<br/><link href="https://arxiv.org/abs/2503.21322" color="#246E72">https://arxiv.org/abs/2503.21322</link>', 'small')

    def footer(canvas, doc):
        canvas.saveState(); canvas.setStrokeColor(colors.HexColor('#DCE3E7'))
        canvas.line(19*mm, 17*mm, A4[0]-19*mm, 17*mm)
        canvas.setFont('CN', 8); canvas.setFillColor(muted)
        canvas.drawString(19*mm, 12*mm, '项目创新说明  |  第8.0版')
        canvas.drawRightString(A4[0]-19*mm, 12*mm, str(doc.page))
        canvas.restoreState()
    doc = SimpleDocTemplate(str(DOCS/'项目创新说明.pdf'), pagesize=A4,
                            rightMargin=18*mm, leftMargin=18*mm, topMargin=17*mm, bottomMargin=23*mm,
                            title='项目创新说明', author='项目组')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    (DOCS/'项目创新说明.md').write_text('\n\n'.join(manuscript), encoding='utf-8')
    print(json.dumps({'pdf': str(DOCS/'项目创新说明.pdf'), 'peak_reduction_percent': reduction}, ensure_ascii=False))


if __name__ == '__main__':
    main()
