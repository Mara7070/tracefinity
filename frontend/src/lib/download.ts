// export artefacts are purged after the retention window (docs/stl-generation.md);
// a 404 means the file expired, so regenerate and retry once instead of
// surfacing the raw json error to the user

export function filenameFromDisposition(header: string | null): string | null {
  if (!header) return null
  const utf8 = /filename\*=utf-8''([^;]+)/i.exec(header)
  if (utf8) {
    try {
      return decodeURIComponent(utf8[1].trim())
    } catch {
      return null
    }
  }
  const plain = /filename="?([^";]+)"?/i.exec(header)
  return plain ? plain[1].trim() : null
}

function fallbackFilename(url: string): string {
  const path = url.split(/[?#]/)[0]
  return path.split('/').pop() || 'download'
}

export async function downloadExport(url: string, regenerate: () => Promise<void>): Promise<void> {
  let response = await fetch(url)
  if (response.status === 404) {
    await regenerate()
    response = await fetch(url)
  }
  if (!response.ok) {
    throw new Error(`export download failed (${response.status})`)
  }
  const blob = await response.blob()
  downloadBlob(blob, filenameFromDisposition(response.headers.get('content-disposition')) ?? fallbackFilename(url))
}

export function downloadBlob(blob: Blob, filename: string): void {
  const objectUrl = URL.createObjectURL(blob)
  try {
    const link = document.createElement('a')
    try {
      link.href = objectUrl
      link.download = filename
      document.body.appendChild(link)
      link.click()
    } finally {
      link.remove()
    }
  } finally {
    URL.revokeObjectURL(objectUrl)
  }
}
