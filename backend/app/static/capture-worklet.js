// Microphone capture: whatever rate the device runs at -> 16 kHz mono PCM16.
//
// Runs on the audio rendering thread, so it does as little as possible: average
// input samples down to the target rate, pack them as 16-bit integers, and post
// a chunk to the main thread every 40 ms. The server's speech detector and the
// transcription model both want 16 kHz, and doing the conversion here means
// 3x less data on the wire than sending a 48 kHz device stream.
//
// The device rate is not resampled by the AudioContext on purpose: connecting a
// MediaStream to a context of a different rate throws in some browsers, so the
// context stays at the device's native rate and the conversion happens here.
//
// Downsampling is a box filter (average of the samples that fall in each output
// sample's span). It is not a high-quality resampler, and for speech recognition
// it does not need to be: the energy that would alias sits above 8 kHz, which
// the microphone chain has already largely removed.

class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const target = options.processorOptions.targetRate;
    this.ratio = sampleRate / target; // input samples per output sample
    this.chunkSamples = Math.round(target * 0.04); // 40 ms
    this.out = new Int16Array(this.chunkSamples);
    this.filled = 0;
    this.phase = 0; // how far through the current output sample we are
    this.acc = 0; // running sum for the current output sample
    this.count = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;

    for (let i = 0; i < channel.length; i++) {
      this.acc += channel[i];
      this.count += 1;
      this.phase += 1;
      if (this.phase >= this.ratio) {
        this.phase -= this.ratio;
        const mean = this.acc / this.count;
        this.acc = 0;
        this.count = 0;
        const clipped = Math.max(-1, Math.min(1, mean));
        this.out[this.filled++] = clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff;
        if (this.filled === this.chunkSamples) {
          this.port.postMessage(this.out.buffer, [this.out.buffer]);
          this.out = new Int16Array(this.chunkSamples);
          this.filled = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor("capture", CaptureProcessor);
