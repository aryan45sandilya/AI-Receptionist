// Speaker output: a queue of audio the main thread hands over, played gaplessly.
//
// Scheduling one AudioBufferSourceNode per chunk works, but every chunk is
// resampled on its own, so the joins between them can click, and there is no way
// to drop audio that is already queued. A worklet with a plain FIFO fixes both:
// samples come out in exact order at the context's rate, and 'flush' empties the
// queue on the next render quantum -- which is what stopping the agent
// mid-sentence needs when the caller talks over it.

class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.offset = 0;
    this.active = false;
    this.port.onmessage = (event) => {
      if (event.data === "flush") {
        this.queue = [];
        this.offset = 0;
      } else {
        this.queue.push(event.data);
      }
    };
  }

  process(_inputs, outputs) {
    const out = outputs[0][0];
    let written = 0;

    while (written < out.length && this.queue.length > 0) {
      const head = this.queue[0];
      const n = Math.min(out.length - written, head.length - this.offset);
      out.set(head.subarray(this.offset, this.offset + n), written);
      written += n;
      this.offset += n;
      if (this.offset >= head.length) {
        this.queue.shift();
        this.offset = 0;
      }
    }
    // Anything not written stays at the zero the output starts as: an underrun
    // is silence, not noise.

    const active = written > 0 || this.queue.length > 0;
    if (active !== this.active) {
      this.active = active;
      this.port.postMessage({ active });
    }
    return true;
  }
}

registerProcessor("playback", PlaybackProcessor);
