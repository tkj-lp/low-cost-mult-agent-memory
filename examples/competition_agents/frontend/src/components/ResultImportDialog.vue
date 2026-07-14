<template>
  <div class="metrics-record-backdrop" @click.self="$emit('close')">
    <section class="metrics-record-dialog" role="dialog" aria-modal="true" aria-label="导入实验结果">
      <header>
        <div>
          <strong>导入实验结果</strong>
          <span>{{ resultFileDirectory || 'run_result' }}</span>
        </div>
        <button type="button" class="metrics-dialog-close" @click="$emit('close')">关闭</button>
      </header>
      <div class="metrics-import-toolbar">
        <button type="button" @click="loadResultFiles">刷新</button>
        <button type="button" @click="selectAllResultFiles">全选</button>
        <button type="button" @click="clearSelectedResultFiles">取消全选</button>
        <span>已选择 {{ selectedImportFiles.length }} 个文件</span>
      </div>
      <div v-if="!resultFiles.length" class="metrics-record-empty">暂无可导入的 JSON 结果文件。</div>
      <div v-else class="metrics-import-tree">
        <section v-for="group in importFileGroups" :key="group.folder" class="metrics-import-folder">
          <label class="metrics-import-folder-row">
            <input
              type="checkbox"
              :checked="isImportFolderSelected(group)"
              @change="toggleImportFolder(group, $event.target.checked)"
            />
            <strong>{{ group.label }}</strong>
            <span>{{ group.files.length }} 个文件 / {{ formatFileSize(group.size) }}</span>
          </label>
          <label
            v-for="file in group.files"
            :key="file.path"
            class="metrics-import-file-row"
          >
            <input
              type="checkbox"
              :value="file.path"
              v-model="selectedImportFiles"
            />
            <span>{{ file.name }}</span>
            <em>{{ file.modifiedLabel }}</em>
            <small>{{ formatFileSize(file.size) }}</small>
          </label>
        </section>
      </div>
      <div class="metrics-import-footer">
        <button type="button" class="metrics-import-cancel" @click="$emit('close')">取消</button>
        <button
          type="button"
          class="metrics-import-submit"
          :disabled="!selectedImportFiles.length || importLoading"
          @click="importSelectedResults"
        >
          {{ importLoading ? '导入中...' : '导入所选' }}
        </button>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'

const RUNNER_RESULT_FILES_URL = 'http://127.0.0.1:8765/api/runner/result-files'
const RUNNER_IMPORT_RESULTS_URL = 'http://127.0.0.1:8765/api/runner/import-results'

const emit = defineEmits(['close', 'imported'])

const resultFiles = ref([])
const selectedImportFiles = ref([])
const resultFileDirectory = ref('')
const importLoading = ref(false)

const importFileGroups = computed(() => {
  const groups = new Map()
  for (const file of resultFiles.value) {
    const folder = file.folder || '未分组'
    if (!groups.has(folder)) groups.set(folder, [])
    groups.get(folder).push(file)
  }
  return [...groups.entries()].map(([folder, files]) => ({
    folder,
    label: folder,
    files,
    size: files.reduce((sum, file) => sum + Number(file.size || 0), 0)
  }))
})

onMounted(() => {
  loadResultFiles()
})

async function loadResultFiles() {
  const response = await fetch(`${RUNNER_RESULT_FILES_URL}?t=${Date.now()}`, { cache: 'no-store' })
  if (!response.ok) return
  const payload = await response.json()
  resultFiles.value = Array.isArray(payload.files) ? payload.files : []
  resultFileDirectory.value = payload.directory || ''
  const validPaths = new Set(resultFiles.value.map((file) => file.path))
  selectedImportFiles.value = selectedImportFiles.value.filter((path) => validPaths.has(path))
}

function selectAllResultFiles() {
  selectedImportFiles.value = resultFiles.value.map((file) => file.path)
}

function clearSelectedResultFiles() {
  selectedImportFiles.value = []
}

function isImportFolderSelected(group) {
  return group.files.length > 0 && group.files.every((file) => selectedImportFiles.value.includes(file.path))
}

function toggleImportFolder(group, checked) {
  const selected = new Set(selectedImportFiles.value)
  for (const file of group.files) {
    if (checked) selected.add(file.path)
    else selected.delete(file.path)
  }
  selectedImportFiles.value = [...selected]
}

async function importSelectedResults() {
  if (!selectedImportFiles.value.length || importLoading.value) return
  importLoading.value = true
  try {
    const response = await fetch(RUNNER_IMPORT_RESULTS_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ files: selectedImportFiles.value })
    })
    if (!response.ok) throw new Error('导入结果失败')
    const payload = await response.json()
    emit('imported', {
      records: Array.isArray(payload.records) ? payload.records : [],
      payloads: Array.isArray(payload.payloads) ? payload.payloads : []
    })
    emit('close')
  } finally {
    importLoading.value = false
  }
}

function formatFileSize(value) {
  const bytes = Number(value || 0)
  if (!bytes) return '0 B'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}
</script>
