const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const path = require('node:path');

(async () => {
  const browser = await chromium.launch({headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe'});
  const page = await browser.newPage({viewport: {width: 1440, height: 1100}});
  const errors = [];
  await page.addInitScript(() => localStorage.setItem('competition-agent-console-config', JSON.stringify({dataset:'topiocqa-ranked-153', taskCount:3})));
  page.on('pageerror', error => errors.push(error.message));
  let phase = 'running';
  const runId = () => phase === 'new' ? 'new-run' : 'test-run';
  const taskNumbers = Array.from({length:10}, (_, index) => index + 1);
  const base = () => ({run_id: runId(), dataset_file:'data/continuous_topiocqa_153_10turns_v1/questions.json', task_count:10,
    experiment_mode:'baseline_text',memory_organization:'text',memory_retrieval_mode:'semantic'});
  const events = () => [
    {...base(),event_type:'run_start'},
    {...base(),event_type:'task_queue',tasks:taskNumbers.map(n=>({task_id:`task-${n}`,task_index:n,task_title:`接口队列测试 ${n}`}))},
    ...(phase === 'new' ? [] : [{...base(),event_type:'task_start',task_id:'task-1',task_index:1,to_agent:'Planner'}]),
    ...(phase === 'failed' ? [{...base(),event_type:'task_failed',task_id:'task-1',task_index:1,return_result:{error:'测试失败'}}] : []),
    ...(phase === 'stopped' ? [{...base(),event_type:'task_done',task_id:'task-1',task_index:1},
      {...base(),event_type:'task_start',task_id:'task-2',task_index:2,to_agent:'Planner'}] : [])
  ];
  await page.route('http://127.0.0.1:8765/api/runner/**', async route => {
    const pathname = new URL(route.request().url()).pathname;
    let data = {};
    if (pathname.endsWith('/status')) data = {runId:runId(),running:['running','new'].includes(phase),status:phase === 'new' ? 'running' : phase,summary:{}};
    if (pathname.endsWith('/events')) data = {events:events()};
    // 故意让当前结果接口晚到一份旧运行，页面不得把它拼进新队列。
    if (pathname.endsWith('/result')) data = {payload:{run_id:'old-run',runs:[{task_results:[1,2,3].map(n=>({task_index:n,status:'completed',final_answer:'旧答案'}))}]}};
    if (pathname.endsWith('/metrics')) data = {records:[]};
    await route.fulfill({json:data});
  });
  await page.goto('http://127.0.0.1:5173/');
  assert.equal(await page.locator('.fixed-count').innerText(), '10');
  assert.match(await page.locator('.dataset-note').innerText(), /第153组完整10轮/);
  await page.locator('.dataset-field select').selectOption('codeact');
  assert.equal(await page.locator('.fixed-count').innerText(), '10');
  await page.locator('.dataset-field select').selectOption('topiocqa-ranked-153');
  await page.screenshot({path:path.resolve(__dirname,'../../runtime/validation_v5_config.png'),fullPage:true});
  await page.getByRole('button',{name:'协作机制',exact:true}).click();
  const statuses = () => page.locator('.task-strip-head b').allTextContents();
  await page.waitForTimeout(1500);
  assert.deepEqual(await statuses(), ['运行中', ...Array(9).fill('未开始')]);
  phase = 'failed'; await page.waitForTimeout(1500);
  assert.deepEqual(await statuses(), ['执行失败', ...Array(9).fill('未开始')]);
  phase = 'stopped'; await page.waitForTimeout(1500);
  assert.deepEqual(await statuses(), ['已完成','已停止', ...Array(8).fill('未开始')]);
  await page.screenshot({path:path.resolve(__dirname,'../../runtime/validation_v5_task_status.png'),fullPage:true});
  phase = 'new'; await page.waitForTimeout(1500);
  assert.deepEqual(await statuses(), Array(10).fill('未开始'));
  await page.getByRole('button',{name:'记忆复用',exact:true}).click();
  await page.waitForTimeout(1500);
  assert.equal(await page.locator('.memory-task-tabs button.done').count(),0);
  assert.equal(await page.locator('.memory-task-tabs button.waiting').count(),10);
  assert.deepEqual(errors,[]);
  console.log('浏览器状态测试通过：运行、失败、停止、新运行、旧结果隔离、记忆页面。');
  await browser.close();
})().catch(error=>{console.error(error);process.exit(1)});
