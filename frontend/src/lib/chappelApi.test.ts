import { afterEach, expect, it, vi } from 'vitest'
import { ApiError, exportChappelTools } from './api'

const options = {
  format: 'step' as const, max_cells_x: 6, max_cells_y: 6,
  gap: 2, depth: 12, clearance: 0.25, inset: 0,
}
afterEach(() => vi.unstubAllGlobals())

it('posts ordered Tools with credentials and returns the binary and server filename', async () => {
  const response = new Response('STEP data', {
    headers: { 'content-disposition': "attachment; filename*=utf-8''caf%C3%A9.step" },
  })
  const fetchMock = vi.fn().mockResolvedValue(response)
  vi.stubGlobal('fetch', fetchMock)
  const result = await exportChappelTools(['z', 'a'], options)
  expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/chappel/tools/export'), {
    method: 'POST', credentials: 'include', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...options, tool_ids: ['z', 'a'] }),
  })
  expect(result.filename).toBe('café.step')
  expect(await result.blob.text()).toBe('STEP data')
})

it('uses the chosen format for a missing filename', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('mesh')))
  expect((await exportChappelTools(['a'], { ...options, format: 'stl' })).filename)
    .toBe('tracefinity-chappel-tools.stl')
})

it('surfaces backend errors through ApiError', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
    detail: { message: 'Tools do not fit', code: 'no_fit' },
  }), { status: 422 })))
  const failure = exportChappelTools(['a'], options)
  await expect(failure).rejects.toBeInstanceOf(ApiError)
  await expect(failure).rejects.toMatchObject({ message: 'Tools do not fit', code: 'no_fit', status: 422 })
})
