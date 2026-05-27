# Competition Agents Frontend

这个目录是基于 `Vue 3 + Element Plus + Vite` 的前端控制台原型，对应
`examples/competition_agents/FRONTEND_CONTRACT.md` 里的页面要求。

## 目标

- 提供两个选择框：运行方式、数据集
- 展示推荐命令映射
- 展示指标卡、通信链路、text/protocol 对比、记忆命中、latent 状态、问答结果表
- 支持导入自定义 JSON，便于后续接真实实验输出

## 启动

```bash
cd examples/competition_agents/frontend
npm install
npm run dev
```

默认开发地址由 `vite.config.js` 配置为：

```text
http://127.0.0.1:4173
```

## 数据接入

当前页面内置了若干示例场景，覆盖契约中的推荐组合。后续如果后端导出真实 JSON，可以直接通过页面里的“导入 JSON”入口加载。

导入对象建议至少包含：

```json
{
  "mode": "latent_hidden",
  "dataset": "hidden_continuous",
  "command": "python ...",
  "summary": {},
  "cases": [
    {
      "group_id": "chain-001",
      "turn_id": "turn-1",
      "question": "....",
      "final_answer": "....",
      "gold": "....",
      "exact_match": true,
      "elapsed_ms": 1000,
      "metrics": {},
      "messages": [],
      "memory_hits": [],
      "latent_states": {}
    }
  ]
}
```

页面只强依赖契约中声明的稳定字段，不依赖 protocol payload 的具体内部结构。
