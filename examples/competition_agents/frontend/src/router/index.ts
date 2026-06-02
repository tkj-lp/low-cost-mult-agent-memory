import { createRouter, createWebHashHistory } from 'vue-router'

import ConfigPanel from '../components/ConfigPanel.vue'
import PipelineRunner from '../components/PipelineRunner.vue'
import MetricDashboard from '../components/MetricDashboard.vue'
import LatentExplorer from '../components/LatentExplorer.vue'

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    {
      path: '/',
      redirect: '/config'
    },
    {
      path: '/config',
      name: 'config',
      component: ConfigPanel
    },
    {
      path: '/pipeline',
      name: 'pipeline',
      component: PipelineRunner
    },
    {
      path: '/metrics',
      name: 'metrics',
      component: MetricDashboard
    },
    {
      path: '/latent',
      name: 'latent',
      component: LatentExplorer
    }
  ]
})

export default router
