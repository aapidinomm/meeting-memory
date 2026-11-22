// meeting-memory — Kiro Crew app page. Plain ESM, no build step.
//
// Records the browser's microphone (and optionally a shared tab's audio), resamples to 16 kHz mono PCM, cuts it
// into overlapping WAV chunks and uploads them one at a time to the app's backend, which transcribes each with
// Amazon Bedrock. Stop triggers the summary and search indexing on the backend.
import { createElement as h, Fragment, useCallback, useEffect, useRef, useState } from 'react'
import { useAppApi } from '@kirocrew/app-sdk'

const BASE = '/api/apps/meeting-memory'
const ACCENT = 'var(--accent, #7c3aed)'
const RATE = 16000
const SILENCE_RMS = 0.003

// ---------------------------------------------------------------- style

const S = {
  page: { maxWidth: '1200px', margin: '0 auto', padding: '20px 20px 40px', color: 'var(--text)', fontSize: '13px', lineHeight: 1.5 },
  card: { background: 'var(--card, var(--bg))', border: '1px solid var(--border)', borderRadius: '10px', padding: '16px', marginBottom: '14px' },
  label: { fontSize: '11px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.04em', color: 'var(--muted)', margin: '0 0 8px' },
  muted: { color: 'var(--muted)', fontSize: '12px' },
  row: { display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' },
  input: { width: '100%', boxSizing: 'border-box', background: 'var(--bg)', color: 'var(--text)', border: '1px solid var(--border)', borderRadius: '6px', padding: '7px 9px', fontSize: '12px' },
  grid: { display: 'grid', gridTemplateColumns: 'minmax(240px, 320px) minmax(0, 1fr)', gap: '14px', alignItems: 'start' },
  ts: { color: 'var(--muted)', fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace', fontSize: '11px', marginRight: '8px', whiteSpace: 'nowrap' },
  scroll: { maxHeight: '420px', overflowY: 'auto' },
}

function btnStyle(kind = 'ghost', disabled = false) {
  const base = { borderRadius: '9999px', fontSize: '12px', fontWeight: 500, padding: '7px 16px', cursor: disabled ? 'default' : 'pointer', whiteSpace: 'nowrap', opacity: disabled ? 0.55 : 1 }
  if (kind === 'primary') return { ...base, background: ACCENT, color: '#fff', border: '1px solid transparent' }
  if (kind === 'danger') return { ...base, background: 'transparent', color: '#b91c1c', border: '1px solid #b91c1c' }
  return { ...base, background: 'transparent', color: 'var(--text)', border: '1px solid var(--border)' }
}
const Btn = ({ kind, disabled, onClick, children, title }) => h('button', { style: btnStyle(kind, disabled), disabled, onClick, title }, children)

const STATUS = {
  recording: ['#fee2e2', '#b91c1c', 'recording'], processing: ['#dbeafe', '#1d4ed8', 'summarizing'],
  done: ['#d1fae5', '#047857', 'done'], error: ['#fef3c7', '#b45309', 'needs attention'],
}
const Pill = ({ status }) => {
  const [bg, fg, text] = STATUS[status] || ['#e5e7eb', '#4b5563', status || '?']
  return h('span', { style: { background: bg, color: fg, padding: '2px 8px', borderRadius: '9999px', fontSize: '11px', fontWeight: 600, whiteSpace: 'nowrap' } }, text)
}

function errText(e) {
  if (!e) return ''
  if (e.body) { try { const j = JSON.parse(e.body); return j.error || e.body } catch (_) { return e.body } }
  return e.message || String(e)
}
const ErrorBox = ({ error, onClose }) => error ? h('div', { style: { ...S.card, borderColor: '#b91c1c', color: '#b91c1c', display: 'flex', justifyContent: 'space-between', gap: '8px' } },
  h('span', null, error), onClose ? h('button', { onClick: onClose, style: { ...btnStyle('ghost'), padding: '2px 10px' } }, 'Dismiss') : null) : null

const fmtTs = (ms) => {
  const s = Math.floor((ms || 0) / 1000)
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, '0'), ss = String(s % 60).padStart(2, '0')
  return s >= 3600 ? `${Math.floor(s / 3600)}:${mm}:${ss}` : `${Math.floor(s / 60)}:${ss}`
}
const fmtDate = (t) => t ? new Date(t * 1000).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) : ''

// Minimal markdown for the summary (headings, bullets, **bold**), built as elements: no innerHTML.
function inline(text) {
  const parts = String(text).split(/(\*\*[^*]+\*\*)/g)
  return parts.map((p, i) => p.startsWith('**') && p.endsWith('**') ? h('strong', { key: i }, p.slice(2, -2)) : p)
}
function Markdown({ text }) {
  const out = []
  let list = []
  const flush = () => { if (list.length) { out.push(h('ul', { key: 'u' + out.length, style: { margin: '4px 0 10px 18px', padding: 0 } }, list)); list = [] } }
  String(text || '').split('\n').forEach((line, i) => {
    if (/^\s*[-*] /.test(line)) { list.push(h('li', { key: i }, inline(line.replace(/^\s*[-*] /, '')))); return }
    flush()
    if (line.startsWith('# ')) out.push(h('h3', { key: i, style: { margin: '0 0 8px', fontSize: '16px' } }, inline(line.slice(2))))
    else if (line.startsWith('## ')) out.push(h('h4', { key: i, style: { margin: '12px 0 4px', fontSize: '13px', color: ACCENT } }, inline(line.slice(3))))
    else if (line.trim()) out.push(h('p', { key: i, style: { margin: '0 0 8px' } }, inline(line)))
  })
  flush()
  return h('div', null, out)
}

// ---------------------------------------------------------------- audio helpers

function encodeWav(samples) {
  const buf = new ArrayBuffer(44 + samples.length * 2)
  const v = new DataView(buf)
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)) }
  str(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); str(8, 'WAVE'); str(12, 'fmt ')
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, RATE, true)
  v.setUint32(28, RATE * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); str(36, 'data')
  v.setUint32(40, samples.length * 2, true)
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return new Blob([buf], { type: 'audio/wav' })
}

function rms(samples) {
  let sum = 0
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i]
  return samples.length ? Math.sqrt(sum / samples.length) : 0
}

// Box-filter downsampler from the AudioContext rate to 16 kHz, stateful across buffers.
function makeResampler(inRate) {
  const ratio = inRate / RATE
  let acc = 0, n = 0, phase = 0
  return (input) => {
    const out = []
    for (let i = 0; i < input.length; i++) {
      acc += input[i]; n++; phase += 1
      if (phase >= ratio) { out.push(acc / n); acc = 0; n = 0; phase -= ratio }
    }
    return out
  }
}

// ---------------------------------------------------------------- recorder

function Recorder({ api, settings, onChanged, onOpen }) {
  const [phase, setPhase] = useState('idle') // idle | starting | recording | paused | stopping
  const [title, setTitle] = useState('')
  const [tabAudio, setTabAudio] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [level, setLevel] = useState(0)
  const [pending, setPending] = useState(0)
  const [segments, setSegments] = useState([])
  const [err, setErr] = useState('')
  const [note, setNote] = useState('')
  const R = useRef(null)
  const liveRef = useRef(null)

  useEffect(() => { if (liveRef.current) liveRef.current.scrollTop = liveRef.current.scrollHeight }, [segments])

  const pump = useCallback(async () => {
    const r = R.current
    if (!r || r.uploading) return
    r.uploading = true
    while (r.queue.length) {
      const item = r.queue[0]
      let ok = false
      for (let attempt = 0; attempt < 3 && !ok; attempt++) {
        try {
          const url = `${BASE}/meetings/${r.id}/chunks?seq=${item.seq}&start_ms=${item.startMs}&duration_ms=${item.durMs}`
          const resp = await api.raw(url, { method: 'POST', headers: { 'Content-Type': 'audio/wav' }, body: item.blob })
          const seg = resp && typeof resp.json === 'function' ? await resp.json() : resp
          if (seg && seg.text) setSegments((s) => [...s, seg])
          ok = true
        } catch (e) {
          if (attempt === 2) { setErr('A chunk failed to transcribe and was skipped: ' + errText(e)) }
          else await new Promise((res) => setTimeout(res, 1500 * (attempt + 1)))
        }
      }
      r.queue.shift()
      setPending(r.queue.length)
    }
    r.uploading = false
  }, [api])

  const emit = useCallback((final) => {
    const r = R.current
    const chunkN = Math.round(settings.chunk_seconds * RATE)
    const overlapN = Math.round(settings.overlap_seconds * RATE)
    while (r.buf.length >= chunkN || (final && r.buf.length >= RATE / 2)) {
      const main = r.buf.splice(0, Math.min(chunkN, r.buf.length))
      const data = r.tail.concat(main)
      const startMs = Math.round(((r.emitted - r.tail.length) / RATE) * 1000)
      const durMs = Math.round((data.length / RATE) * 1000)
      const seq = r.seq++
      r.emitted += main.length
      r.tail = overlapN ? main.slice(-overlapN) : []
      if (rms(main) < SILENCE_RMS) continue // skip silent chunks: no Bedrock call for dead air
      r.queue.push({ seq, startMs, durMs, blob: encodeWav(data) })
      setPending(r.queue.length)
      pump()
    }
  }, [settings, pump])

  const teardown = () => {
    const r = R.current
    if (!r) return
    try { r.proc && r.proc.disconnect() } catch (_) { }
    for (const s of r.streams) for (const t of s.getTracks()) t.stop()
    try { r.ctx && r.ctx.close() } catch (_) { }
    clearInterval(r.timer)
  }

  const start = async () => {
    setErr(''); setNote(''); setSegments([]); setElapsed(0); setPhase('starting')
    const streams = []
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('This page cannot use the microphone. Open the dashboard over https or http://localhost (for a remote host, an SSH port-forward).')
      }
      streams.push(await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } }))
      if (tabAudio) {
        const disp = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true })
        if (!disp.getAudioTracks().length) {
          for (const t of disp.getTracks()) t.stop()
          setNote('The shared tab or screen had no audio, so only the microphone is recorded. Tick "Share tab audio" in the browser picker.')
        } else streams.push(disp)
      }
      const meeting = await api.post(BASE + '/meetings', { title })
      const ctx = new AudioContext()
      const proc = ctx.createScriptProcessor(4096, 1, 1)
      for (const s of streams) if (s.getAudioTracks().length) ctx.createMediaStreamSource(s).connect(proc)
      proc.connect(ctx.destination) // a ScriptProcessor only runs while connected; it writes silence
      const resample = makeResampler(ctx.sampleRate)
      R.current = { id: meeting.id, ctx, proc, streams, buf: [], tail: [], emitted: 0, seq: 0, queue: [], uploading: false, paused: false, timer: null }
      proc.onaudioprocess = (ev) => {
        const r = R.current
        if (!r || r.paused) return
        const input = ev.inputBuffer.getChannelData(0)
        const out = resample(input)
        for (let i = 0; i < out.length; i++) r.buf.push(out[i])
        r.level = rms(input)
        emit(false)
      }
      R.current.timer = setInterval(() => {
        const r = R.current
        if (!r) return
        setElapsed(Math.round(((r.emitted + r.buf.length) / RATE) * 1000))
        setLevel(r.paused ? 0 : Math.min(1, (r.level || 0) * 8))
      }, 250)
      setPhase('recording')
      onChanged()
    } catch (e) {
      for (const s of streams) for (const t of s.getTracks()) t.stop()
      setErr(errText(e)); setPhase('idle')
    }
  }

  const togglePause = () => {
    const r = R.current
    if (!r) return
    r.paused = !r.paused
    setPhase(r.paused ? 'paused' : 'recording')
  }

  const stop = useCallback(async () => {
    const r = R.current
    if (!r) return
    setPhase('stopping')
    r.paused = true
    emit(true)
    teardown()
    while (r.uploading || r.queue.length) await new Promise((res) => setTimeout(res, 300))
    try { await api.post(`${BASE}/meetings/${r.id}/stop`, {}) } catch (e) { setErr(errText(e)) }
    const id = r.id
    R.current = null
    setPhase('idle'); setLevel(0)
    onChanged(); onOpen(id)
  }, [api, emit, onChanged, onOpen])

  // Leaving the page ends the capture; stop cleanly so what was said is kept and summarized. The ref keeps this
  // effect mount-only: re-running it whenever `stop` changes identity would stop a live recording.
  const stopRef = useRef(stop)
  stopRef.current = stop
  useEffect(() => {
    const warn = (e) => { if (R.current) { e.preventDefault(); e.returnValue = '' } }
    window.addEventListener('beforeunload', warn)
    return () => { window.removeEventListener('beforeunload', warn); if (R.current) stopRef.current() }
  }, [])

  const busy = phase === 'starting' || phase === 'stopping'
  const live = phase === 'recording' || phase === 'paused'
  return h('div', { style: S.card },
    h('div', { style: S.label }, 'Record'),
    h(ErrorBox, { error: err, onClose: () => setErr('') }),
    note ? h('div', { style: { ...S.muted, marginBottom: '8px' } }, note) : null,
    !live && !busy ? h('div', { style: { ...S.row, marginBottom: '10px' } },
      h('input', { style: { ...S.input, flex: '1 1 260px', width: 'auto' }, placeholder: 'Meeting title (optional: the summary names it otherwise)', value: title, onChange: (e) => setTitle(e.target.value) }),
      h('label', { style: { ...S.row, ...S.muted, gap: '6px' }, title: 'Also record the audio of a browser tab (Chime, Zoom or Meet in a tab), so remote voices are captured even with headphones' },
        h('input', { type: 'checkbox', checked: tabAudio, onChange: (e) => setTabAudio(e.target.checked) }), 'Also capture a tab\u2019s audio'),
    ) : null,
    h('div', { style: S.row },
      !live ? h(Btn, { kind: 'primary', disabled: busy || !settings.aws_profile, onClick: start }, phase === 'starting' ? 'Starting\u2026' : phase === 'stopping' ? 'Finishing\u2026' : '\u25cf Start recording') : null,
      live ? h(Btn, { onClick: togglePause }, phase === 'paused' ? 'Resume' : 'Pause') : null,
      live ? h(Btn, { kind: 'danger', onClick: stop }, 'Stop and summarize') : null,
      live || busy ? h('span', { style: { fontFamily: 'ui-monospace, monospace', fontSize: '14px', fontWeight: 600 } }, fmtTs(elapsed)) : null,
      live ? h('span', { title: 'input level', style: { display: 'inline-block', width: '90px', height: '6px', borderRadius: '3px', background: 'var(--border)', overflow: 'hidden' } },
        h('span', { style: { display: 'block', height: '100%', width: `${Math.round(level * 100)}%`, background: phase === 'paused' ? 'var(--muted)' : '#047857', transition: 'width 0.2s' } })) : null,
      live || busy ? h('span', { style: S.muted }, pending ? `${pending} chunk${pending > 1 ? 's' : ''} transcribing` : (phase === 'paused' ? 'paused' : 'listening')) : null,
    ),
    live ? h('div', { style: { ...S.muted, marginTop: '8px' } }, `Keep this page open while recording. Text appears about ${settings.chunk_seconds} s behind the speaker. Get everyone\u2019s consent before recording.`) : null,
    segments.length ? h('div', { ref: liveRef, style: { ...S.scroll, marginTop: '12px', maxHeight: '260px', borderTop: '1px solid var(--border)', paddingTop: '8px' } },
      segments.map((s) => h('div', { key: s.seq, style: { marginBottom: '6px' } }, h('span', { style: S.ts }, fmtTs(s.start_ms)), s.text))) : null,
  )
}

// ---------------------------------------------------------------- meeting detail

function MeetingDetail({ api, id, onChanged, onDeleted }) {
  const [m, setM] = useState(null)
  const [err, setErr] = useState('')
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')
  const [titleDraft, setTitleDraft] = useState('')
  const [tab, setTab] = useState('summary')

  const load = useCallback(async () => {
    try { const r = await api.get(`${BASE}/meetings/${id}`); setM(r); setErr('') } catch (e) { setErr(errText(e)) }
  }, [api, id])
  useEffect(() => { setM(null); setEditing(false); load() }, [load])
  useEffect(() => {
    if (!m || m.meta.status !== 'processing') return
    const t = setInterval(load, 4000)
    return () => clearInterval(t)
  }, [m, load])

  if (err) return h(ErrorBox, { error: err })
  if (!m) return h('div', { style: { ...S.card, ...S.muted } }, 'Loading\u2026')
  const meta = m.meta
  const act = async (fn) => { try { await fn(); await load(); onChanged() } catch (e) { setErr(errText(e)) } }
  const regenerate = () => act(() => api.post(`${BASE}/meetings/${id}/summarize`, {}))
  const saveSummary = () => act(async () => { await api.put(`${BASE}/meetings/${id}`, { summary_md: draft }); setEditing(false) })
  const saveTitle = () => act(() => api.put(`${BASE}/meetings/${id}`, { title: titleDraft }))
  const remove = async () => {
    if (!window.confirm('Delete this meeting, its audio, transcript and summary? This cannot be undone.')) return
    try { await api.post(`${BASE}/meetings/${id}/delete`, {}); onDeleted() } catch (e) { setErr(errText(e)) }
  }
  const turns = m.turns || []
  const renameSpeaker = (sid, current) => {
    const name = window.prompt(`Name for ${sid}`, current === sid ? '' : current)
    if (name === null) return
    act(() => api.put(`${BASE}/meetings/${id}`, { speaker_names: { [sid]: name.trim() } }))
  }
  const download = () => {
    const text = (turns.length ? turns.map((t) => `[${fmtTs(t.start_ms)}] ${t.speaker}: ${t.text}`)
      : (m.transcript || []).map((s) => `[${fmtTs(s.start_ms)}] ${s.text}`)).join('\n')
    const url = URL.createObjectURL(new Blob([`${meta.title}\n${fmtDate(meta.created_at)}\n\n${text}\n`], { type: 'text/plain' }))
    const a = document.createElement('a')
    a.href = url; a.download = `${meta.id}-transcript.txt`; a.click()
    setTimeout(() => URL.revokeObjectURL(url), 2000)
  }
  const tabBtn = (key, label) => h('button', { onClick: () => setTab(key), style: { ...btnStyle(tab === key ? 'primary' : 'ghost'), padding: '4px 12px' } }, label)

  return h('div', { style: S.card },
    h('div', { style: { ...S.row, justifyContent: 'space-between', marginBottom: '10px' } },
      h('div', { style: { ...S.row, flex: '1 1 300px' } },
        h('input', { style: { ...S.input, fontSize: '15px', fontWeight: 600, flex: '1 1 240px', width: 'auto' }, defaultValue: meta.title, key: meta.title,
          onChange: (e) => setTitleDraft(e.target.value), onBlur: () => { if (titleDraft && titleDraft !== meta.title) saveTitle() } }),
        h(Pill, { status: meta.status })),
      h('div', { style: S.row },
        h(Btn, { onClick: download, disabled: !(m.transcript || []).length }, 'Download transcript'),
        h(Btn, { onClick: regenerate, disabled: meta.status === 'processing' || meta.status === 'recording' }, 'Regenerate summary'),
        h(Btn, { kind: 'danger', onClick: remove, disabled: meta.status === 'recording' }, 'Delete'))),
    h('div', { style: { ...S.muted, marginBottom: '10px' } }, `${fmtDate(meta.created_at)} \u00b7 ${fmtTs(meta.duration_ms)} recorded`),
    meta.error ? h('div', { style: { color: '#b45309', marginBottom: '10px' } }, meta.error) : null,
    meta.speakers_status && meta.speakers_status !== 'ok' && meta.speakers_status !== 'off'
      ? h('div', { style: { ...S.muted, marginBottom: '10px' } }, `Speaker labels ${meta.speakers_status}`) : null,
    h('div', { style: { ...S.row, marginBottom: '10px' } }, tabBtn('summary', 'Summary'),
      turns.length ? tabBtn('speakers', `Speakers (${meta.speakers || speakerIds(turns).length})`) : null,
      tabBtn('transcript', `Transcript (${(m.transcript || []).length})`)),
    tab === 'speakers' ? h(SpeakerView, { turns, onRename: renameSpeaker }) :
    tab === 'summary' ? (meta.status === 'processing' ? h('div', { style: S.muted }, 'Writing the summary and indexing the transcript\u2026')
      : editing ? h(Fragment, null,
        h('textarea', { style: { ...S.input, minHeight: '320px', fontFamily: 'ui-monospace, monospace' }, value: draft, onChange: (e) => setDraft(e.target.value) }),
        h('div', { style: { ...S.row, marginTop: '8px' } }, h(Btn, { kind: 'primary', onClick: saveSummary }, 'Save'), h(Btn, { onClick: () => setEditing(false) }, 'Cancel')))
        : h(Fragment, null,
          m.summary_md ? h(Markdown, { text: m.summary_md }) : h('div', { style: S.muted }, 'No summary yet.'),
          m.summary_md ? h('div', { style: { marginTop: '8px' } }, h(Btn, { onClick: () => { setDraft(m.summary_md); setEditing(true) } }, 'Edit summary')) : null))
      : h('div', { style: S.scroll }, (m.transcript || []).length ? m.transcript.map((s) => h('div', { key: s.seq, style: { marginBottom: '6px' } }, h('span', { style: S.ts }, fmtTs(s.start_ms)), s.text))
        : h('div', { style: S.muted }, 'No speech was transcribed.')),
  )
}

// ---------------------------------------------------------------- speakers

// Distinct hues that read on light and dark themes; text stays var(--text).
const SPEAKER_HUES = [210, 25, 145, 280, 350, 55, 185, 320]
const speakerIds = (turns) => [...new Set(turns.map((t) => t.speaker_id))]

function SpeakerView({ turns, onRename }) {
  const ids = speakerIds(turns)
  const hue = (sid) => SPEAKER_HUES[ids.indexOf(sid) % SPEAKER_HUES.length]
  const nameOf = {}
  const talk = {}
  for (const t of turns) {
    nameOf[t.speaker_id] = t.speaker
    talk[t.speaker_id] = (talk[t.speaker_id] || 0) + Math.max(0, (t.end_ms || 0) - (t.start_ms || 0))
  }
  const total = Object.values(talk).reduce((a, b) => a + b, 0) || 1
  const chip = (sid) => h('button', {
    key: sid, onClick: () => onRename(sid, nameOf[sid]), title: 'Click to rename',
    style: { border: `1px solid hsl(${hue(sid)} 60% 50%)`, borderLeft: `6px solid hsl(${hue(sid)} 60% 50%)`, background: 'transparent', color: 'var(--text)', borderRadius: '6px', padding: '3px 10px', fontSize: '12px', cursor: 'pointer' },
  }, `${nameOf[sid]} \u00b7 ${Math.round((100 * talk[sid]) / total)}%`)
  return h(Fragment, null,
    h('div', { style: { ...S.row, marginBottom: '8px' } }, ids.map(chip)),
    h('div', { style: { ...S.muted, fontSize: '11px', marginBottom: '8px' } }, 'Speakers are found from their voices. Click a name to rename it everywhere. Words near a change of speaker can land with the wrong person.'),
    h('div', { style: S.scroll }, turns.map((t, i) => h('div', { key: i, style: { marginBottom: '8px', paddingLeft: '8px', borderLeft: `3px solid hsl(${hue(t.speaker_id)} 60% 50%)` } },
      h('div', { style: { fontSize: '12px', fontWeight: 600 } }, h('span', { style: S.ts }, fmtTs(t.start_ms)), t.speaker),
      h('div', null, t.text)))))
}

// ---------------------------------------------------------------- search and ask

function SearchPanel({ api, onOpen }) {
  const [q, setQ] = useState('')
  const [res, setRes] = useState(null)
  const [busy, setBusy] = useState('')
  const [err, setErr] = useState('')
  const run = async (mode) => {
    if (!q.trim()) return
    setBusy(mode); setErr('')
    try {
      setRes(mode === 'ask' ? { mode, ...(await api.post(BASE + '/ask', { question: q })) } : { mode, ...(await api.post(BASE + '/search', { query: q })) })
    } catch (e) { setErr(errText(e)) } finally { setBusy('') }
  }
  return h('div', { style: S.card },
    h('div', { style: S.label }, 'Search your meetings'),
    h('div', { style: S.row },
      h('input', { style: { ...S.input, flex: '1 1 260px', width: 'auto' }, placeholder: 'e.g. what did we decide about the reward registry?', value: q,
        onChange: (e) => setQ(e.target.value), onKeyDown: (e) => { if (e.key === 'Enter') run('search') } }),
      h(Btn, { onClick: () => run('search'), disabled: !!busy }, busy === 'search' ? 'Searching\u2026' : 'Search'),
      h(Btn, { kind: 'primary', onClick: () => run('ask'), disabled: !!busy }, busy === 'ask' ? 'Thinking\u2026' : 'Ask')),
    h(ErrorBox, { error: err }),
    res && res.notes && res.notes.length ? h('div', { style: { ...S.muted, marginTop: '8px' } }, res.notes.join(' ')) : null,
    res && res.mode === 'ask' ? h('div', { style: { marginTop: '10px' } },
      h('div', { style: { whiteSpace: 'pre-wrap', marginBottom: '8px' } }, res.answer),
      (res.sources || []).map((s) => h('div', { key: s.n, style: S.muted }, `[${s.n}] `,
        h('a', { href: '#', onClick: (e) => { e.preventDefault(); onOpen(s.meeting_id) }, style: { color: ACCENT } }, s.title), s.at ? ` at ${s.at}` : ' (summary)'))) : null,
    res && res.mode === 'search' ? h('div', { style: { marginTop: '10px' } }, (res.hits || []).length ? res.hits.map((hit, i) => h('div', { key: i, style: { padding: '8px 0', borderTop: '1px solid var(--border)' } },
      h('div', { style: S.row }, h('a', { href: '#', onClick: (e) => { e.preventDefault(); onOpen(hit.meeting_id) }, style: { color: ACCENT, fontWeight: 600 } }, hit.title),
        h('span', { style: S.muted }, hit.kind === 'summary' ? 'summary' : `at ${hit.at}`)),
      h('div', { style: { fontSize: '12px', marginTop: '2px' } }, hit.text.length > 400 ? hit.text.slice(0, 400) + '\u2026' : hit.text)))
      : h('div', { style: S.muted }, 'No matches.')) : null,
  )
}

// ---------------------------------------------------------------- settings

const FIELDS = [
  ['aws_profile', 'AWS profile', 'A named profile from ~/.aws/config on the machine running Kiro Crew'],
  ['aws_region', 'Bedrock region', 'e.g. us-west-2'],
  ['stt_model', 'Speech-to-text model', 'Pick from the list; scores come from eval/stt/compare.py'],
  ['llm_model', 'Summary / answer model', 'Model id or inference profile, e.g. us.anthropic.claude-sonnet-4-6'],
  ['embed_model', 'Embedding model', 'amazon.titan-embed-text-v2:0 or cohere.embed-v4:0 (us.cohere.embed-v4:0)'],
  ['chunk_seconds', 'Chunk length (s)', 'Audio per transcription call, 5 to 60'],
  ['overlap_seconds', 'Chunk overlap (s)', 'Repeated audio at each boundary so no word is cut, 0 to 5'],
  ['language_hint', 'Language hint', 'Optional, e.g. English'],
]

function SettingsPanel({ api, settings, onSaved, firstRun }) {
  const [form, setForm] = useState(() => ({ ...settings, vocabulary: (settings.vocabulary || []).join('\n') }))
  const [busy, setBusy] = useState('')
  const [err, setErr] = useState('')
  const [test, setTest] = useState(null)
  const [stt, setStt] = useState(null)
  useEffect(() => { api.get(BASE + '/stt-models').then(setStt).catch(() => setStt(null)) }, [api])
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }))
  const modelLabel = (md) => {
    const bits = [md.label]
    if (md.wer != null) bits.push(`${(100 * md.wer).toFixed(1)}% words wrong`)
    bits.push(md.kind === 'local' ? '$0' : (md.usd_per_hour != null ? `$${md.usd_per_hour.toFixed(3)}/hour` : 'Bedrock, billed'))
    if (md.seconds_per_audio_minute != null) bits.push(`${md.seconds_per_audio_minute}s per audio min`)
    if (stt && stt.recommended === md.id) bits.push('best')
    if (!md.available) bits.push('needs scripts/setup_engine.py')
    return bits.join(' \u00b7 ')
  }
  const field = (k, help) => k === 'stt_model' && stt && stt.models
    ? h('select', { style: S.input, value: form[k] ?? '', onChange: set(k) },
      (stt.models.some((md) => md.id === form[k]) ? [] : [h('option', { key: form[k], value: form[k] }, form[k])])
        .concat(stt.models.map((md) => h('option', { key: md.id, value: md.id }, modelLabel(md)))))
    : h('input', { style: S.input, value: form[k] ?? '', onChange: set(k), placeholder: help })
  const save = async (andTest) => {
    setBusy(andTest ? 'test' : 'save'); setErr(''); setTest(null)
    try {
      const patch = {}
      for (const [k] of FIELDS) patch[k] = form[k]
      patch.vocabulary = form.vocabulary
      if (!firstRun) patch.speaker_labels = form.speaker_labels !== false
      if (patch.stt_model === settings.stt_model) delete patch.stt_model // unchanged: keep auto-selection on
      const saved = await api.put(BASE + '/settings', patch)
      if (andTest) setTest(await api.post(BASE + '/settings/test', {}))
      onSaved(saved)
    } catch (e) { setErr(errText(e)) } finally { setBusy('') }
  }
  const shown = firstRun ? FIELDS.slice(0, 2) : FIELDS
  return h('div', { style: S.card },
    h('div', { style: S.label }, firstRun ? 'Set up Bedrock access' : 'Settings'),
    firstRun ? h('p', { style: { ...S.muted, marginTop: 0 } }, 'meeting-memory calls Amazon Bedrock in your own AWS account. Enter the name of an AWS profile that the Kiro Crew gateway can use (only the name is stored, never credentials) and the region. You can change this later in Settings.') : null,
    h(ErrorBox, { error: err }),
    h('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '10px' } },
      shown.map(([k, label, help]) => h('label', { key: k, style: { display: 'block' } },
        h('div', { style: { fontSize: '12px', fontWeight: 600, marginBottom: '3px' } }, label),
        field(k, help),
        h('div', { style: { ...S.muted, fontSize: '11px' } }, help)))),
    !firstRun ? h('label', { style: { ...S.row, ...S.muted, gap: '6px', marginTop: '10px' } },
      h('input', { type: 'checkbox', checked: form.speaker_labels !== false, onChange: (e) => setForm((f) => ({ ...f, speaker_labels: e.target.checked })) }),
      'Label who spoke when, on Stop (runs on this machine)') : null,
    !firstRun ? h('label', { style: { display: 'block', marginTop: '10px' } },
      h('div', { style: { fontSize: '12px', fontWeight: 600, marginBottom: '3px' } }, 'Vocabulary'),
      h('textarea', { style: { ...S.input, minHeight: '80px' }, value: form.vocabulary, onChange: set('vocabulary'), placeholder: 'One per line: names, acronyms and product terms to spell correctly' })) : null,
    h('div', { style: { ...S.row, marginTop: '12px' } },
      h(Btn, { kind: 'primary', onClick: () => save(true), disabled: !!busy }, busy === 'test' ? 'Testing\u2026' : 'Save and test'),
      !firstRun ? h(Btn, { onClick: () => save(false), disabled: !!busy }, 'Save') : null),
    test ? h('div', { style: { marginTop: '10px', fontSize: '12px' } },
      h('div', null, test.ok ? '\u2705 All checks passed' : '\u26a0\ufe0f Some checks failed', test.account ? ` \u00b7 account ${test.account}` : ''),
      Object.entries(test.checks || {}).map(([k, v]) => h('div', { key: k, style: { color: v === 'ok' ? '#047857' : '#b91c1c' } }, `${k}: ${v}`))) : null,
  )
}

// ---------------------------------------------------------------- page

export default function MeetingMemory() {
  const api = useAppApi()
  const [settings, setSettings] = useState(null)
  const [meetings, setMeetings] = useState([])
  const [selected, setSelected] = useState(null)
  const [view, setView] = useState('meetings')
  const [err, setErr] = useState('')

  const loadMeetings = useCallback(async () => {
    try { const r = await api.get(BASE + '/meetings'); setMeetings(r.meetings || []) } catch (e) { setErr(errText(e)) }
  }, [api])
  useEffect(() => {
    api.get(BASE + '/settings').then(setSettings).catch((e) => setErr(errText(e)))
    loadMeetings()
  }, [api, loadMeetings])
  useEffect(() => {
    if (!meetings.some((m) => m.status === 'processing')) return
    const t = setInterval(loadMeetings, 5000)
    return () => clearInterval(t)
  }, [meetings, loadMeetings])

  const open = useCallback((id) => { setSelected(id); setView('meetings') }, [])
  if (!settings) return h('div', { style: S.page }, err ? h(ErrorBox, { error: err }) : h('div', { style: S.muted }, 'Loading\u2026'))
  const firstRun = !settings.aws_profile

  return h('div', { style: S.page },
    h('div', { style: { ...S.row, justifyContent: 'space-between', marginBottom: '16px' } },
      h('div', { style: S.row },
        h('h2', { style: { margin: 0, fontSize: '18px' } }, 'Meeting Memory'),
        h('span', { style: { background: '#e8d5f5', color: '#7c3aed', padding: '2px 8px', borderRadius: '9999px', fontSize: '10px', fontWeight: 600 } },
          firstRun ? 'not set up' : `${settings.aws_profile} \u00b7 ${settings.aws_region}`)),
      !firstRun ? h('div', { style: S.row },
        h(Btn, { kind: view === 'meetings' ? 'primary' : 'ghost', onClick: () => setView('meetings') }, 'Meetings'),
        h(Btn, { kind: view === 'settings' ? 'primary' : 'ghost', onClick: () => setView('settings') }, 'Settings')) : null),
    h(ErrorBox, { error: err, onClose: () => setErr('') }),
    firstRun ? h(SettingsPanel, { api, settings, firstRun: true, onSaved: setSettings }) : null,
    // The recorder stays mounted across the Meetings/Settings switch so a live recording is not cut off.
    !firstRun ? h(Recorder, { api, settings, onChanged: loadMeetings, onOpen: open }) : null,
    firstRun ? null
      : view === 'settings' ? h(SettingsPanel, { api, settings, onSaved: setSettings })
        : h(Fragment, null,
          h(SearchPanel, { api, onOpen: open }),
          h('div', { style: S.grid },
            h('div', { style: S.card },
              h('div', { style: S.label }, `Meetings (${meetings.length})`),
              meetings.length ? h('div', { style: { ...S.scroll, maxHeight: '640px' } }, meetings.map((m) => h('div', {
                key: m.id, onClick: () => setSelected(m.id),
                style: { padding: '8px', borderRadius: '6px', cursor: 'pointer', marginBottom: '4px', background: selected === m.id ? 'var(--border)' : 'transparent' },
              },
                h('div', { style: { ...S.row, justifyContent: 'space-between' } }, h('span', { style: { fontWeight: 600 } }, m.title), h(Pill, { status: m.status })),
                h('div', { style: S.muted }, `${fmtDate(m.created_at)} \u00b7 ${fmtTs(m.duration_ms)}`),
                m.tldr ? h('div', { style: { fontSize: '12px', marginTop: '2px' } }, m.tldr.length > 140 ? m.tldr.slice(0, 140) + '\u2026' : m.tldr) : null)))
                : h('div', { style: S.muted }, 'No meetings yet. Press Start recording.')),
            selected ? h(MeetingDetail, { api, id: selected, onChanged: loadMeetings, onDeleted: () => { setSelected(null); loadMeetings() } })
              : h('div', { style: { ...S.card, ...S.muted } }, 'Select a meeting to see its summary and transcript.'))),
  )
}
