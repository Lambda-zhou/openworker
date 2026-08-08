//! A local, offline speech-to-text engine.
//!
//! This crate deliberately has no Tauri, UI, clipboard, or global-shortcut dependency. Hosts
//! own their own UX and permission flows; they use [`Dictation`] for microphone capture,
//! model provisioning, and final transcription.

use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc, Mutex,
    },
    thread,
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    SampleFormat, Stream, StreamConfig,
};
use serde::Serialize;
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineSenseVoiceModelConfig};

/// SenseVoice multi-language model (int8 quantized) — ~226 MB, supports zh/en/ja/ko/yue.
pub const DEFAULT_MODEL_FILE: &str = "model.int8.onnx";
pub const DEFAULT_TOKENS_FILE: &str = "tokens.txt";
pub const DEFAULT_MODEL_URL: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/resolve/main/model.int8.onnx";
pub const DEFAULT_TOKENS_URL: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/resolve/main/tokens.txt";
pub const DEFAULT_MODEL_BYTES: u64 = 237_115_547;
pub const DEFAULT_TOKENS_BYTES: u64 = 315_894;
const SENSEVOICE_SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Clone, Serialize)]
pub struct DictationStatus {
    pub recording: bool,
    pub model_installed: bool,
    pub model_verified: bool,
    pub test_passed: bool,
    pub download_in_progress: bool,
    pub model_name: &'static str,
    pub model_bytes: u64,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct DownloadProgress {
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
}

struct Recording {
    stream: Stream,
    samples: Arc<Mutex<Vec<f32>>>,
    sample_rate: u32,
}

/// A reusable single-microphone dictation session manager.
///
/// It records only while a host has explicitly started a session; audio is held in memory for
/// that session and is never persisted. The downloaded recognition model is the only data kept
/// under `model_dir`.
pub struct Dictation {
    model_path: PathBuf,
    tokens_path: PathBuf,
    verified_marker_path: PathBuf,
    ready_marker_path: PathBuf,
    commands: Sender<Command>,
    recording: Arc<Mutex<bool>>,
    // Live handle onto the in-flight recording's sample buffer (set by the capture worker
    // for the duration of a session) so hosts can meter input loudness for UI feedback.
    live: Arc<Mutex<Option<(Arc<Mutex<Vec<f32>>>, u32)>>>,
    download_in_progress: AtomicBool,
    cancel_download: AtomicBool,
}

enum Command {
    Start(Sender<Result<(), String>>),
    Stop(Sender<Result<RecordedAudio, String>>),
    Cancel(Sender<()>),
}

struct RecordedAudio {
    samples: Vec<f32>,
    sample_rate: u32,
}

impl Dictation {
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        // CPAL's CoreAudio stream is intentionally !Send. Keep it on one dedicated owner thread
        // rather than unsafely forcing it through Tauri's Send + Sync application state.
        let (commands, receiver) = mpsc::channel();
        let recording = Arc::new(Mutex::new(false));
        let live = Arc::new(Mutex::new(None));
        let worker_recording = recording.clone();
        let worker_live = live.clone();
        thread::spawn(move || capture_worker(receiver, worker_recording, worker_live));
        let model_dir = model_dir.into();
        let model_path = model_dir.join(DEFAULT_MODEL_FILE);
        let tokens_path = model_dir.join(DEFAULT_TOKENS_FILE);
        Self {
            verified_marker_path: model_path.with_extension("bin.verified"),
            ready_marker_path: model_path.with_extension("bin.ready"),
            model_path,
            tokens_path,
            commands,
            recording,
            live,
            download_in_progress: AtomicBool::new(false),
            cancel_download: AtomicBool::new(false),
        }
    }

    pub fn status(&self) -> DictationStatus {
        let model_installed = self.model_path.is_file();
        let model_verified = model_installed
            && self.tokens_path.is_file()
            && model_verification_marker_matches(&self.model_path, &self.verified_marker_path);
        DictationStatus {
            recording: self.recording.lock().map(|r| *r).unwrap_or(false),
            model_installed,
            model_verified,
            test_passed: model_verified && self.ready_marker_path.is_file(),
            download_in_progress: self.download_in_progress.load(Ordering::SeqCst),
            model_name: "SenseVoice (zh/en/ja/ko/yue)",
            model_bytes: DEFAULT_MODEL_BYTES,
        }
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Downloads the default model atomically. Hosts should call this only after an explicit
    /// user action because it is a sizeable download.
    pub fn install_default_model(&self) -> Result<(), String> {
        self.install_default_model_with_progress(|_| {})
    }

    /// Downloads and verifies the default model atomically, reporting byte progress to the host.
    /// A canceled/failed transfer never replaces a previously verified model.
    pub fn install_default_model_with_progress(
        &self,
        mut on_progress: impl FnMut(DownloadProgress),
    ) -> Result<(), String> {
        if self.status().model_verified {
            return Ok(());
        }
        if self
            .download_in_progress
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err("The local voice model is already downloading.".to_owned());
        }
        self.cancel_download.store(false, Ordering::SeqCst);

        let result = (|| {
            let parent = self
                .model_path
                .parent()
                .ok_or_else(|| "Could not determine the local model directory.".to_owned())?;
            fs::create_dir_all(parent)
                .map_err(|e| format!("Could not create model directory: {e}"))?;

            let agent = ureq::AgentBuilder::new()
                .timeout_connect(std::time::Duration::from_secs(30))
                .timeout_read(std::time::Duration::from_secs(30))
                .build();

            let total_bytes = DEFAULT_MODEL_BYTES + DEFAULT_TOKENS_BYTES;
            let mut cumulative = 0_u64;

            // --- download model ---
            let model_part = self.model_path.with_extension("onnx.part");
            download_file_to(
                &agent,
                DEFAULT_MODEL_URL,
                &model_part,
                DEFAULT_MODEL_BYTES,
                &mut cumulative,
                total_bytes,
                &self.cancel_download,
                &mut on_progress,
            )?;
            verify_model_file(&model_part)?;
            if self.model_path.exists() {
                fs::remove_file(&self.model_path)
                    .map_err(|e| format!("Could not replace the local voice model: {e}"))?;
            }
            fs::rename(&model_part, &self.model_path)
                .map_err(|e| format!("Could not install the local voice model: {e}"))?;

            // --- download tokens ---
            let tokens_part = self.tokens_path.with_extension("txt.part");
            download_file_to(
                &agent,
                DEFAULT_TOKENS_URL,
                &tokens_part,
                DEFAULT_TOKENS_BYTES,
                &mut cumulative,
                total_bytes,
                &self.cancel_download,
                &mut on_progress,
            )?;
            verify_tokens_file(&tokens_part)?;
            if self.tokens_path.exists() {
                fs::remove_file(&self.tokens_path)
                    .map_err(|e| format!("Could not replace the tokens file: {e}"))?;
            }
            fs::rename(&tokens_part, &self.tokens_path)
                .map_err(|e| format!("Could not install the tokens file: {e}"))?;

            write_verification_marker(&self.model_path, &self.verified_marker_path)?;
            let _ = fs::remove_file(&self.ready_marker_path);
            on_progress(DownloadProgress {
                downloaded_bytes: cumulative,
                total_bytes,
            });
            Ok(())
        })();

        self.download_in_progress.store(false, Ordering::SeqCst);
        self.cancel_download.store(false, Ordering::SeqCst);
        result
    }

    /// Verifies an already-installed model (including installs made by older app versions).
    pub fn verify_default_model(&self) -> Result<(), String> {
        verify_model_file(&self.model_path)?;
        verify_tokens_file(&self.tokens_path)?;
        write_verification_marker(&self.model_path, &self.verified_marker_path)
    }

    pub fn cancel_model_download(&self) {
        self.cancel_download.store(true, Ordering::SeqCst);
    }

    pub fn mark_test_passed(&self) -> Result<(), String> {
        if !self.status().model_verified {
            return Err("Verify the local voice model before testing it.".to_owned());
        }
        fs::write(&self.ready_marker_path, b"ready")
            .map_err(|e| format!("Could not save the voice input test result: {e}"))
    }

    pub fn delete_default_model(&self) -> Result<(), String> {
        self.cancel_model_download();
        self.cancel();
        for path in [
            self.model_path.clone(),
            self.model_path.with_extension("onnx.part"),
            self.tokens_path.clone(),
            self.tokens_path.with_extension("txt.part"),
            self.verified_marker_path.clone(),
            self.ready_marker_path.clone(),
        ] {
            if path.exists() {
                fs::remove_file(&path)
                    .map_err(|e| format!("Could not remove {}: {e}", path.display()))?;
            }
        }
        Ok(())
    }

    /// Begins microphone capture. A host must call [`stop_and_transcribe`](Self::stop_and_transcribe)
    /// or [`cancel`](Self::cancel) before a new recording can start.
    pub fn start(&self) -> Result<(), String> {
        if !self.status().model_verified {
            return Err("Set up and verify Voice Input in Settings first.".to_owned());
        }
        let (reply, result) = mpsc::channel();
        self.commands
            .send(Command::Start(reply))
            .map_err(|_| "Dictation is unavailable because its audio worker stopped.".to_owned())?;
        result
            .recv()
            .map_err(|_| "Dictation is unavailable because its audio worker stopped.".to_owned())?
    }

    /// Stops capture and returns a final local transcript. This is intentionally synchronous so
    /// hosts can run it off their UI thread and decide how to present completion/error states.
    pub fn stop_and_transcribe(&self) -> Result<String, String> {
        let (reply, result) = mpsc::channel();
        self.commands
            .send(Command::Stop(reply))
            .map_err(|_| "Dictation is unavailable because its audio worker stopped.".to_owned())?;
        let RecordedAudio {
            samples,
            sample_rate,
        } = result.recv().map_err(|_| {
            "Dictation is unavailable because its audio worker stopped.".to_owned()
        })??;
        if samples.len() < (sample_rate as usize / 4) {
            return Ok(String::new());
        }
        transcribe(&self.model_path, &self.tokens_path, &resample_mono(&samples, sample_rate))
    }

    /// Instantaneous input loudness of the in-flight recording, 0.0..=1.0 — RMS over the
    /// most recent ~100ms, scaled so conversational speech spans most of the range. 0.0
    /// while not recording. Cheap enough to poll at UI frame-ish rates.
    pub fn input_level(&self) -> f32 {
        let live = match self.live.lock() {
            Ok(guard) => guard,
            Err(_) => return 0.0,
        };
        let Some((samples, sample_rate)) = live.as_ref() else {
            return 0.0;
        };
        let Ok(samples) = samples.lock() else {
            return 0.0;
        };
        let window = (*sample_rate as usize / 10).max(1);
        let tail = &samples[samples.len().saturating_sub(window)..];
        if tail.is_empty() {
            return 0.0;
        }
        let mean_square: f32 = tail.iter().map(|s| s * s).sum::<f32>() / tail.len() as f32;
        (mean_square.sqrt() * 8.0).clamp(0.0, 1.0)
    }

    /// Discards the current in-memory recording without retaining or transcribing it.
    pub fn cancel(&self) {
        let (reply, done) = mpsc::channel();
        if self.commands.send(Command::Cancel(reply)).is_ok() {
            let _ = done.recv();
        }
    }
}

fn verify_model_file(path: &Path) -> Result<(), String> {
    let metadata =
        fs::metadata(path).map_err(|e| format!("Could not read the local voice model: {e}"))?;
    if metadata.len() != DEFAULT_MODEL_BYTES {
        return Err(format!(
            "The local voice model is incomplete ({} of {} bytes).",
            metadata.len(),
            DEFAULT_MODEL_BYTES
        ));
    }
    Ok(())
}

fn verify_tokens_file(path: &Path) -> Result<(), String> {
    let metadata = fs::metadata(path)
        .map_err(|e| format!("Could not read the tokens file: {e}"))?;
    if metadata.len() != DEFAULT_TOKENS_BYTES {
        return Err(format!(
            "The tokens file is incomplete ({} of {} bytes).",
            metadata.len(),
            DEFAULT_TOKENS_BYTES
        ));
    }
    Ok(())
}

fn download_file_to(
    agent: &ureq::Agent,
    url: &str,
    dest: &Path,
    expected_bytes: u64,
    cumulative: &mut u64,
    total_bytes: u64,
    cancel_flag: &AtomicBool,
    on_progress: &mut impl FnMut(DownloadProgress),
) -> Result<(), String> {
    let response = agent
        .get(url)
        .call()
        .map_err(|e| format!("Could not download {url}: {e}"))?;
    let mut input = response.into_reader();
    let mut output = fs::File::create(dest)
        .map_err(|e| format!("Could not save {}: {e}", dest.display()))?;
    let mut downloaded = 0_u64;
    let mut last_reported = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    on_progress(DownloadProgress {
        downloaded_bytes: *cumulative,
        total_bytes,
    });
    loop {
        if cancel_flag.load(Ordering::SeqCst) {
            drop(output);
            let _ = fs::remove_file(dest);
            return Err("Voice model download canceled.".to_owned());
        }
        let count = input
            .read(&mut buffer)
            .map_err(|e| format!("Could not download {url}: {e}"))?;
        if count == 0 {
            break;
        }
        output
            .write_all(&buffer[..count])
            .map_err(|e| format!("Could not save {}: {e}", dest.display()))?;
        downloaded += count as u64;
        *cumulative += count as u64;
        if downloaded.saturating_sub(last_reported) >= 512 * 1024
            || downloaded == expected_bytes
        {
            last_reported = downloaded;
            on_progress(DownloadProgress {
                downloaded_bytes: *cumulative,
                total_bytes,
            });
        }
    }
    output
        .flush()
        .map_err(|e| format!("Could not finish saving {}: {e}", dest.display()))?;
    Ok(())
}

fn model_modified_millis(path: &Path) -> Option<u128> {
    fs::metadata(path)
        .ok()?
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_millis())
}

fn write_verification_marker(model_path: &Path, marker_path: &Path) -> Result<(), String> {
    let modified = model_modified_millis(model_path)
        .ok_or_else(|| "Could not read the installed voice model timestamp.".to_owned())?;
    fs::write(marker_path, format!("size:{DEFAULT_MODEL_BYTES}\n{modified}\n"))
        .map_err(|e| format!("Could not record voice model verification: {e}"))
}

fn model_verification_marker_matches(model_path: &Path, marker_path: &Path) -> bool {
    let Ok(metadata) = fs::metadata(model_path) else {
        return false;
    };
    if metadata.len() != DEFAULT_MODEL_BYTES {
        return false;
    }
    let Ok(marker) = fs::read_to_string(marker_path) else {
        return false;
    };
    let mut lines = marker.lines();
    let size_matches = lines
        .next()
        .and_then(|s| s.strip_prefix("size:"))
        .and_then(|v| v.parse::<u64>().ok())
        == Some(DEFAULT_MODEL_BYTES);
    let marker_modified = lines.next().and_then(|value| value.parse::<u128>().ok());
    size_matches && marker_modified == model_modified_millis(model_path)
}

fn capture_worker(
    receiver: Receiver<Command>,
    recording_status: Arc<Mutex<bool>>,
    live: Arc<Mutex<Option<(Arc<Mutex<Vec<f32>>>, u32)>>>,
) {
    let mut recording: Option<Recording> = None;
    let set_live = |value: Option<(Arc<Mutex<Vec<f32>>>, u32)>| {
        if let Ok(mut guard) = live.lock() {
            *guard = value;
        }
    };
    for command in receiver {
        match command {
            Command::Start(reply) => {
                if recording.is_some() {
                    let _ = reply.send(Err("Dictation is already recording.".to_owned()));
                    continue;
                }
                match start_recording() {
                    Ok(next) => {
                        set_live(Some((next.samples.clone(), next.sample_rate)));
                        recording = Some(next);
                        if let Ok(mut active) = recording_status.lock() {
                            *active = true;
                        }
                        let _ = reply.send(Ok(()));
                    }
                    Err(error) => {
                        let _ = reply.send(Err(error));
                    }
                }
            }
            Command::Stop(reply) => {
                set_live(None);
                let result = recording
                    .take()
                    .ok_or_else(|| "Dictation is not recording.".to_owned())
                    .and_then(finish_recording);
                if let Ok(mut active) = recording_status.lock() {
                    *active = false;
                }
                let _ = reply.send(result);
            }
            Command::Cancel(reply) => {
                set_live(None);
                recording.take();
                if let Ok(mut active) = recording_status.lock() {
                    *active = false;
                }
                let _ = reply.send(());
            }
        }
    }
}

fn start_recording() -> Result<Recording, String> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| "No microphone is available. Check your Mac sound settings.".to_owned())?;
    let supported = device
        .default_input_config()
        .map_err(|e| format!("Could not open the microphone: {e}"))?;
    let config: StreamConfig = supported.clone().into();
    let samples = Arc::new(Mutex::new(Vec::new()));
    let stream = build_stream(&device, &config, supported.sample_format(), samples.clone())?;
    stream
        .play()
        .map_err(|e| format!("Could not start microphone recording: {e}"))?;
    Ok(Recording {
        stream,
        samples,
        sample_rate: config.sample_rate.0,
    })
}

fn finish_recording(recording: Recording) -> Result<RecordedAudio, String> {
    let Recording {
        stream,
        samples,
        sample_rate,
    } = recording;
    drop(stream);
    let samples = samples
        .lock()
        .map_err(|_| "Could not read the recorded audio.".to_owned())?
        .clone();
    Ok(RecordedAudio {
        samples,
        sample_rate,
    })
}

fn build_stream(
    device: &cpal::Device,
    config: &StreamConfig,
    sample_format: SampleFormat,
    samples: Arc<Mutex<Vec<f32>>>,
) -> Result<Stream, String> {
    let channels = config.channels as usize;
    let on_error = |error| eprintln!("[ocw-stt] microphone stream error: {error}");
    match sample_format {
        SampleFormat::F32 => device
            .build_input_stream(
                config,
                move |data: &[f32], _| append_frames(&samples, data, channels, |sample| sample),
                on_error,
                None,
            )
            .map_err(|e| format!("Could not create microphone stream: {e}")),
        SampleFormat::I16 => device
            .build_input_stream(
                config,
                move |data: &[i16], _| {
                    append_frames(&samples, data, channels, |sample| {
                        sample as f32 / i16::MAX as f32
                    })
                },
                on_error,
                None,
            )
            .map_err(|e| format!("Could not create microphone stream: {e}")),
        SampleFormat::U16 => device
            .build_input_stream(
                config,
                move |data: &[u16], _| {
                    append_frames(&samples, data, channels, |sample| {
                        (sample as f32 / u16::MAX as f32) * 2.0 - 1.0
                    })
                },
                on_error,
                None,
            )
            .map_err(|e| format!("Could not create microphone stream: {e}")),
        other => Err(format!("Unsupported microphone sample format: {other:?}")),
    }
}

fn append_frames<T>(
    target: &Arc<Mutex<Vec<f32>>>,
    data: &[T],
    channels: usize,
    convert: impl Fn(T) -> f32,
) where
    T: Copy,
{
    let Ok(mut output) = target.lock() else {
        return;
    };
    output.reserve(data.len() / channels.max(1));
    for frame in data.chunks(channels.max(1)) {
        let sum: f32 = frame.iter().copied().map(&convert).sum();
        output.push(sum / frame.len() as f32);
    }
}

fn resample_mono(input: &[f32], source_rate: u32) -> Vec<f32> {
    if source_rate == SENSEVOICE_SAMPLE_RATE {
        return input.to_vec();
    }
    let output_len =
        (input.len() as u64 * SENSEVOICE_SAMPLE_RATE as u64 / source_rate as u64) as usize;
    let ratio = source_rate as f64 / SENSEVOICE_SAMPLE_RATE as f64;
    (0..output_len)
        .map(|i| {
            let position = i as f64 * ratio;
            let left = position.floor() as usize;
            let right = (left + 1).min(input.len().saturating_sub(1));
            let fraction = (position - left as f64) as f32;
            input[left] * (1.0 - fraction) + input[right] * fraction
        })
        .collect()
}

fn transcribe(model_path: &Path, tokens_path: &Path, samples: &[f32]) -> Result<String, String> {
    if !model_path.is_file() {
        return Err("The local voice model is not installed yet.".to_owned());
    }
    if !tokens_path.is_file() {
        return Err("The tokens file is not installed yet.".to_owned());
    }
    let model_path_str = model_path
        .to_str()
        .ok_or_else(|| "The local voice model path is not valid text.".to_owned())?;
    let tokens_path_str = tokens_path
        .to_str()
        .ok_or_else(|| "The tokens file path is not valid text.".to_owned())?;

    let sense_voice_config = OfflineSenseVoiceModelConfig {
        model: Some(model_path_str.to_owned()),
        language: None, // auto-detect
        use_itn: true,
    };

    let config = OfflineRecognizerConfig {
        model_config: sherpa_onnx::OfflineModelConfig {
            sense_voice: sense_voice_config,
            tokens: Some(tokens_path_str.to_owned()),
            ..Default::default()
        },
        ..Default::default()
    };

    let recognizer = OfflineRecognizer::create(&config)
        .ok_or_else(|| "Could not load the local voice model.".to_owned())?;
    let stream = recognizer.create_stream();
    stream.accept_waveform(SENSEVOICE_SAMPLE_RATE, samples);

    recognizer.decode(&stream);
    let result = stream.get_result();
    Ok(result.text().trim().to_owned())
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::{
        resample_mono, write_verification_marker, Dictation, DEFAULT_MODEL_BYTES,
        DEFAULT_MODEL_FILE, DEFAULT_TOKENS_FILE,
    };

    #[test]
    fn resampling_preserves_a_16khz_stream() {
        let input = vec![0.0, 0.5, -0.5];
        assert_eq!(resample_mono(&input, 16_000), input);
    }

    #[test]
    fn resampling_converts_duration() {
        let input = vec![0.0; 48_000];
        assert_eq!(resample_mono(&input, 48_000).len(), 16_000);
    }

    #[test]
    fn default_model_size_matches_the_published_sense_voice_int8_artifact() {
        assert_eq!(DEFAULT_MODEL_BYTES, 237_115_547);
    }

    #[test]
    fn readiness_requires_a_verified_model_and_persists_after_a_test() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ocw-stt-readiness-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        let model = dir.join(DEFAULT_MODEL_FILE);
        fs::File::create(&model)
            .unwrap()
            .set_len(DEFAULT_MODEL_BYTES)
            .unwrap();
        let tokens = dir.join(DEFAULT_TOKENS_FILE);
        fs::File::create(&tokens)
            .unwrap()
            .set_len(DEFAULT_TOKENS_BYTES)
            .unwrap();
        let dictation = Dictation::new(&dir);
        assert!(!dictation.status().model_verified);
        write_verification_marker(&model, &dictation.verified_marker_path).unwrap();
        assert!(dictation.status().model_verified);
        assert!(!dictation.status().test_passed);
        dictation.mark_test_passed().unwrap();
        assert!(dictation.status().test_passed);
        dictation.delete_default_model().unwrap();
        assert!(!dictation.status().model_installed);
        drop(dictation);
        fs::remove_dir_all(dir).unwrap();
    }
}
