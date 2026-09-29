# Generate test speech with the voice built into Windows, so the voice pipeline
# can be exercised without a microphone.
#
#   powershell -ExecutionPolicy Bypass -File scripts\make_test_speech.ps1
#
# Writes 16 kHz mono 16-bit WAV files -- exactly what the browser sends -- into
# scratch\speech\. Feed them to the server with:
#
#   python -m scripts.voice_client scratch\speech\01_booking.wav ...
#
# These are synthetic voices, so they say nothing about how the transcription
# model copes with a real accent or a noisy line. They do prove the plumbing.

param(
    [string]$OutDir = "scratch\speech"
)

Add-Type -AssemblyName System.Speech
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$lines = [ordered]@{
    "01_booking"    = "Hi, I need a dental cleaning next Tuesday evening."
    "02_confirm"    = "Yes, the first one works for me."
    "03_name"       = "My name is Asha Menon."
    "04_human"      = "Actually, can I please speak to a real person?"
    "05_medical"    = "My tooth has been hurting for three days. What should I take for the pain?"
    "06_vague"      = "I want to book a cleaning, sometime whenever you have space."
}

$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono)

$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
Write-Host "Voice: $($voice.Voice.Name)"

foreach ($name in $lines.Keys) {
    $path = Join-Path (Resolve-Path $OutDir) "$name.wav"
    $voice.SetOutputToWaveFile($path, $format)
    $voice.Speak($lines[$name])
    $voice.SetOutputToNull()
    Write-Host ("{0,-14} {1,8:N0} bytes  {2}" -f $name, (Get-Item $path).Length, $lines[$name])
}
$voice.Dispose()
