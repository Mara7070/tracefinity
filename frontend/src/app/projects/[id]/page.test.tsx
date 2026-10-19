// @vitest-environment jsdom
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import ProjectPage from './page'
import * as api from '@/lib/api'
import { downloadBlob } from '@/lib/download'
import type { BinProject, ToolSummary } from '@/types'

vi.mock('@/lib/download', async importOriginal => ({
  ...await importOriginal<typeof import('@/lib/download')>(),
  downloadBlob: vi.fn(),
}))

vi.mock('next/navigation', () => ({
  useParams: () => ({ id: 'p' }), useRouter: () => ({ push: vi.fn() }),
}))
vi.mock('@/hooks/useTheme', () => ({ useTheme: () => ({ theme: 'dark' }) }))
vi.mock('@/components/BinConfigurator', () => ({ BinConfigurator: () => null }))
vi.mock('@/components/ToolSummaryItem', () => ({
  ToolSummaryItem: ({ tool }: { tool: ToolSummary }) => <span>{tool.name}</span>,
  ToolSummaryButton: () => null,
}))
vi.mock('@/lib/api', async importOriginal => ({
  ...await importOriginal<typeof import('@/lib/api')>(),
  getProject: vi.fn(), listTools: vi.fn(), listBins: vi.fn(), listProjects: vi.fn(),
  getProjectHealth: vi.fn(), exportChappelTools: vi.fn(), createProjectBin: vi.fn(),
}))
const project: BinProject = {
  id: 'p', name: 'Workshop', description: null, status: 'active', tool_ids: ['z', 'a'],
  bin_ids: [], placed_tool_ids: [], unplaced_tool_ids: ['z', 'a'], sketches: [],
  default_bin_config: null, notes: null, created_at: null, updated_at: null,
}
beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  vi.mocked(api.getProject).mockResolvedValue(project)
  vi.mocked(api.listTools).mockResolvedValue(['z', 'a'].map(id => ({
    id, name: id, project_ids: ['p'], tags: [], points: [], interior_rings: [],
  } as unknown as ToolSummary)))
  vi.mocked(api.listBins).mockResolvedValue([])
  vi.mocked(api.listProjects).mockResolvedValue([])
  vi.mocked(api.getProjectHealth).mockResolvedValue({ issues: [] } as never)
  vi.mocked(downloadBlob).mockReset()
})
afterEach(() => { cleanup(); vi.restoreAllMocks() })

async function openExport() {
  render(<ProjectPage />)
  await screen.findByText('Workshop')
  expect((screen.getByRole('button', { name: 'Export Chappel' }) as HTMLButtonElement).disabled).toBe(true)
  expect((screen.getByRole('button', { name: 'Create bin' }) as HTMLButtonElement).disabled).toBe(true)
  const projectSelectAll = screen.getAllByRole('button', { name: 'Select all' })
    .find(button => !(button as HTMLButtonElement).disabled)
  expect(projectSelectAll).toBeDefined()
  fireEvent.click(projectSelectAll!)
  fireEvent.click(screen.getByRole('button', { name: 'Export Chappel (2)' }))
}

it('shares selection with Create bin, sends defaults and delegates the download', async () => {
  const blob = new Blob(['STEP'])
  vi.mocked(api.exportChappelTools).mockResolvedValue({ blob, filename: 'server.step' })
  vi.mocked(api.createProjectBin).mockRejectedValue(new Error('bin failure'))
  await openExport()
  fireEvent.click(screen.getByRole('button', { name: 'Download STEP' }))
  await waitFor(() => expect(downloadBlob).toHaveBeenCalledWith(blob, 'server.step'))
  expect(api.exportChappelTools).toHaveBeenCalledWith(['z', 'a'], {
    format: 'step', max_cells_x: 6, max_cells_y: 6, gap: 2, depth: 12, clearance: 0.25, inset: 0,
  })
  expect(downloadBlob).toHaveBeenCalledTimes(1)
  fireEvent.click(screen.getByRole('button', { name: 'Create bin (2)' }))
  await screen.findByText('bin failure')
  expect(api.createProjectBin).toHaveBeenCalledWith('p', expect.objectContaining({ tool_ids: ['z', 'a'] }))
  const projectSelectNone = screen.getAllByRole('button', { name: 'Select none' })
    .find(button => !(button as HTMLButtonElement).disabled)
  expect(projectSelectNone).toBeDefined()
  fireEvent.click(projectSelectNone!)
  expect((screen.getByRole('button', { name: 'Download STEP' }) as HTMLButtonElement).disabled).toBe(true)
})

it.each([new api.ApiError('No fit', 422), new Error('Network failed')])('blocks duplicate exports and displays %s', async error => {
  let reject!: (error: Error) => void
  vi.mocked(api.exportChappelTools).mockImplementation(() => new Promise((_, fail) => { reject = fail }))
  await openExport()
  fireEvent.change(screen.getByLabelText('Format'), { target: { value: 'stl' } })
  fireEvent.change(screen.getByLabelText('Max cells X'), { target: { value: '4' } })
  fireEvent.click(screen.getByRole('button', { name: 'Download STL' }))
  const busy = screen.getByRole('button', { name: 'Exporting…' }) as HTMLButtonElement
  expect(busy.disabled).toBe(true)
  fireEvent.click(busy)
  expect(api.exportChappelTools).toHaveBeenCalledTimes(1)
  expect(api.exportChappelTools).toHaveBeenCalledWith(['z', 'a'], expect.objectContaining({ format: 'stl', max_cells_x: 4 }))
  reject(error)
  await screen.findByText(error.message)
  expect((screen.getByRole('button', { name: 'Download STL' }) as HTMLButtonElement).disabled).toBe(false)
  expect(downloadBlob).not.toHaveBeenCalled()
})

it('preserves individual selection order and displays shared download failures', async () => {
  vi.mocked(api.exportChappelTools).mockResolvedValue({ blob: new Blob(['STEP']), filename: 'insert.step' })
  render(<ProjectPage />)
  await screen.findByText('Workshop')
  // Choose the second project Tool first, then the first.
  fireEvent.click(screen.getAllByTitle('Select Tool')[1])
  fireEvent.click(screen.getAllByTitle('Select Tool')[0])
  fireEvent.click(screen.getByRole('button', { name: 'Export Chappel (2)' }))
  vi.mocked(downloadBlob).mockImplementation(() => { throw new Error('Download failed') })
  fireEvent.click(screen.getByRole('button', { name: 'Download STEP' }))
  await screen.findByText('Download failed')
  expect(api.exportChappelTools).toHaveBeenCalledWith(['a', 'z'], expect.any(Object))
  expect(downloadBlob).toHaveBeenCalledWith(expect.any(Blob), 'insert.step')
})
