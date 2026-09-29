// Browser side of the voice demo.
//
// microphone -> capture worklet (16 kHz PCM16) -> WebSocket -> server
// server -> WebSocket -> resample -> playback worklet -> speaker
//
// Everything intelligent happens on the server: it finds utterances, transcribes,
// thinks and speaks. This file only moves audio and shows what the server says.

const $ = (id) => document.getElementById(id);
const els = {
  business: $("business"),
  clock: $("clock"),
  start: $("start"),
  hangup: $("hangup"),
  state: $("state"),
  notice: $("notice"),
  transcript: $("transcript"),
  latency: $("latency"),
};

const call = {
  ws: null,
  ctx: null,
  stream: null,
  capture: null,
  playback: null,
  resampler: null,
  agentLine: null, // the transcript row the current reply streams into
  turns: 0,
};

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

function setState(text, mode = "") {
  els.state.textContent = text;
  els.state.dataset.mode = mode;
}

function showNotice(text) {
  els.notice.textContent = text;
  els.notice.hidden = !text;
}

function addLine(who, text, kind) {
  const empty = els.transcript.querySelector(".empty");
  if (empty) empty.remove();

  const li = document.createElement("li");
  li.className = kind;
  const label = document.createElement("span");
  label.className = "who";
  label.textContent = who;
  const body = document.createElement("span");
  body.textContent = text;
  li.append(label, body);
  els.transcript.append(li);
  li.scrollIntoView({ block: "nearest" });
  return body;
}

function addLatencyRow(m) {
  const empty = els.latency.querySelector(".empty");
  if (empty) empty.remove();

  call.turns += 1;
  const tr = document.createElement("tr");
  const cell = (value) => {
    const td = document.createElement("td");
    td.textContent = value ?? "none";
    if (value == null) td.className = "none";
    return td;
  };
  const turn = document.createElement("td");
  turn.textContent = String(call.turns);
  tr.append(
    turn,
    cell(m.endpoint_ms),
    cell(m.stt_final_ms),
    cell(m.llm_first_token_ms),
    cell(m.total_ms),
  );
  els.latency.append(tr);
}

// ---------------------------------------------------------------------------
// Playback: resample the voice's rate to the device's, without clicking
// ---------------------------------------------------------------------------

// Linear interpolation that carries its position across chunks. Resampling each
// 40 ms chunk on its own would round differently at every join and put a tick
// between them; tracking one continuous position makes the joins invisible.
class Resampler {
  constructor(inRate, outRate) {
    this.ratio = inRate / outRate;
    this.k = 0; // next output sample index, counted from the start of the clip
    this.base = 0; // input samples consumed before the current chunk
    this.prev = 0; // last input sample of the previous chunk
  }

  push(int16) {
    const out = [];
    const n = int16.length;
    for (;;) {
      const p = this.k * this.ratio - this.base;
      const i = Math.floor(p);
      if (i + 1 >= n) break; // the next sample lives in the next chunk
      const a = i < 0 ? this.prev : int16[i];
      const b = int16[i + 1];
      out.push((a + (b - a) * (p - i)) / 32768);
      this.k += 1;
    }
    this.prev = int16[n - 1];
    this.base += n;
    return Float32Array.from(out);
  }
}

// ---------------------------------------------------------------------------
// Server messages
// ---------------------------------------------------------------------------

function onEvent(message) {
  switch (message.type) {
    case "ready":
      els.business.textContent = message.business;
      els.clock.textContent = message.local_time;
      setState("Greeting you", "");
      addLine("Agent", message.greeting, "agent");
      break;

    case "speech_started":
      setState("Hearing you");
      break;

    case "speech_ended":
      setState(message.used ? "Thinking" : "Listening", message.used ? "" : "listening");
      break;

    case "transcript":
      if (message.discarded) {
        setState("Listening", "listening");
      } else {
        addLine("You", message.text, "you");
        call.agentLine = null;
      }
      break;

    case "agent_delta":
      // The reply appears as it is written, ahead of the audio catching up.
      if (!call.agentLine) call.agentLine = addLine("Agent", "", "agent");
      call.agentLine.textContent += message.text;
      break;

    case "audio_start":
      call.resampler = new Resampler(message.sample_rate, call.ctx.sampleRate);
      setState("Speaking");
      break;

    case "turn_done":
      addLatencyRow(message);
      call.agentLine = null;
      break;

    case "tts_error":
      // Not hidden: the reply is on screen, but the caller heard nothing.
      showNotice(
        "The voice could not speak: " + message.message +
        " If it mentions terms, accept them once in the Groq console for the Orpheus model."
      );
      break;

    case "turn_error":
      showNotice("That turn failed: " + message.message);
      setState("Listening", "listening");
      break;

    case "handoff":
      addLine("Call", "Transferred to a colleague: " + message.reason, "system");
      break;

    case "fatal":
      showNotice(message.message);
      setState("Call failed", "error");
      break;

    case "listening":
      // The server has stopped discarding audio: it is the caller's turn.
      setState("Listening", "listening");
      break;

    case "audio_end":
      break;
  }
}

function onAudio(buffer) {
  if (!call.resampler || !call.playback) return;
  const samples = call.resampler.push(new Int16Array(buffer));
  if (samples.length) call.playback.port.postMessage(samples, [samples.buffer]);
}

// ---------------------------------------------------------------------------
// Call lifecycle
// ---------------------------------------------------------------------------

async function startCall() {
  showNotice("");
  els.start.disabled = true;
  setState("Connecting");

  try {
    call.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        // The browser's own echo cancellation is what stops the agent hearing
        // itself through the speakers. Not optional for a call.
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });
  } catch (error) {
    showNotice(
      "Microphone unavailable (" + error.name + "). Allow microphone access for this page and try again."
    );
    setState("Not connected", "error");
    els.start.disabled = false;
    return;
  }

  call.ctx = new AudioContext({ latencyHint: "interactive" });
  await call.ctx.resume();
  await call.ctx.audioWorklet.addModule("capture-worklet.js");
  await call.ctx.audioWorklet.addModule("playback-worklet.js");

  call.playback = new AudioWorkletNode(call.ctx, "playback", { numberOfInputs: 0 });
  call.playback.connect(call.ctx.destination);
  call.playback.port.onmessage = (event) => {
    // The queue running dry is the honest signal that the agent has finished
    // talking -- the server can only say when it stopped *sending*.
    if (event.data.active === false && call.ws && call.ws.readyState === WebSocket.OPEN) {
      setState("Listening", "listening");
    }
  };

  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${scheme}://${location.host}/ws/voice`);
  ws.binaryType = "arraybuffer";
  call.ws = ws;

  ws.onopen = () => {
    const source = call.ctx.createMediaStreamSource(call.stream);
    call.capture = new AudioWorkletNode(call.ctx, "capture", {
      processorOptions: { targetRate: 16000 },
      numberOfOutputs: 0,
    });
    call.capture.port.onmessage = (event) => {
      // Chunks already in flight when the call ends still arrive after the
      // socket has been cleared, so this cannot assume it exists.
      if (call.ws && call.ws.readyState === WebSocket.OPEN) call.ws.send(event.data);
    };
    source.connect(call.capture);
    els.hangup.disabled = false;
  };

  ws.onmessage = (event) => {
    if (typeof event.data === "string") onEvent(JSON.parse(event.data));
    else onAudio(event.data);
  };

  // A socket that has been closed keeps firing events for a moment. If the
  // caller has already started a new call, those must not end it.
  ws.onclose = () => {
    if (call.ws === ws) endCall("Call ended");
  };
  ws.onerror = () => {
    if (call.ws === ws) showNotice("Could not reach the server. Is it running?");
  };
}

function endCall(label) {
  if (call.ws && call.ws.readyState === WebSocket.OPEN) {
    call.ws.send(JSON.stringify({ type: "hangup" }));
    call.ws.close();
  }
  if (call.stream) call.stream.getTracks().forEach((track) => track.stop());
  if (call.capture) {
    call.capture.port.onmessage = null;
    call.capture.disconnect();
  }
  if (call.ctx) call.ctx.close();
  Object.assign(call, { ws: null, ctx: null, stream: null, capture: null, playback: null, resampler: null });

  els.start.disabled = false;
  els.hangup.disabled = true;
  setState(label);
}

els.start.addEventListener("click", startCall);
els.hangup.addEventListener("click", () => endCall("Call ended"));

// Exposed for automated checks: lets a test read state and replace the microphone.
window.__demo = { call, els };
