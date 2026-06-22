# Whisper CoreML (Apple Neural Engine) Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the Whisper encoder on the Apple Neural Engine via CoreML to make transcription 2-3× faster, with automatic CPU fallback when no CoreML encoder is present.

**Architecture:** Enable `WHISPER_COREML` in the CMake-generated `libwhisper` static library so whisper.cpp 1.7.5 will load a `<model>-encoder.mlmodelc` next to each `.bin` and run the encoder on the ANE. Wire the extra CoreML object code and `CoreML.framework`/`Foundation.framework` into the app target. Extend `WhisperModelManager` so that, when a Whisper model is downloaded, the matching CoreML encoder bundle is also downloaded, unzipped, and renamed to the exact path whisper.cpp expects. `WHISPER_COREML_ALLOW_FALLBACK` guarantees that a missing/failed encoder silently falls back to the existing CPU path — never a regression.

**Tech Stack:** whisper.cpp 1.7.5 (vendored under `libwhisper/whisper.cpp`), CMake → Xcode generator, Swift/SwiftUI macOS app, CoreML, `URLSession`, `Foundation` archive/unzip.

## Global Constraints

- Platform: macOS 14.0+ (`CMAKE_OSX_DEPLOYMENT_TARGET "14.0"`), Apple Silicon is the primary target; Intel build must still succeed and simply never produce a CoreML encoder (CPU path only). CoreML acceleration is Apple-Silicon-only.
- whisper.cpp version is pinned at 1.7.5 — do not bump it as part of this work.
- Fallback is mandatory: `WHISPER_COREML_ALLOW_FALLBACK` must be ON so a missing or incompatible `.mlmodelc` degrades to CPU instead of failing transcription.
- CoreML encoder path contract (whisper.cpp `whisper_get_coreml_path_encoder`, `libwhisper/whisper.cpp/src/whisper.cpp:3344-3359`): for a model file `X.bin`, the encoder must be at the sibling path `X-encoder.mlmodelc` (literally `.bin` replaced by `-encoder.mlmodelc`). The three catalog quant variants share ONE upstream encoder but each needs its own correctly-named copy.
- Models directory: `~/Library/Application Support/<bundleId>/whisper-models/` (`WhisperModelManager.modelsDirectory`).
- Existing FP16/quant catalog and the `#1` relabeling work are explicitly OUT OF SCOPE.
- Do not break the existing `run.sh` dev build or `make_release.sh` two-arch release flow.

---

## File Structure

- `libwhisper/CMakeLists.txt` — turn on the two CoreML CMake options (build config).
- `OpenSuperWhisper.xcodeproj/project.pbxproj` — link `libwhisper.coreml.a`, `CoreML.framework`, `Foundation.framework` into the app target; declare the `whisper.coreml` sub-target dependency.
- `OpenSuperWhisper/CoreMLModel.swift` (new) — pure helper: maps a model `.bin` filename to its encoder bundle name, its upstream zip URL, and the on-disk `.mlmodelc` target path. Fully unit-testable, no I/O.
- `OpenSuperWhisper/WhisperModelManager.swift` — add `downloadCoreMLEncoder(...)` (download zip → unzip → rename to target) and `isCoreMLEncoderPresent(forModel:)`.
- `OpenSuperWhisper/Settings.swift` — add `coreMLEncoderURL` to `SettingsDownloadableModel`, populate it for the three turbo entries, and trigger encoder download after the `.bin` download; surface a "Neural Engine" status row.
- `OpenSuperWhisper/Onboarding/OnboardingView.swift` — same encoder-download trigger in the onboarding download path.
- `OpenSuperWhisperTests/CoreMLModelTests.swift` (new) — unit tests for the path/URL mapping helper.
- `docs/` / `README.md` — note Neural Engine acceleration + first-run compile behavior.

---

## Task 1: Enable CoreML in CMake and determine linkage (build spike)

**Files:**
- Modify: `libwhisper/CMakeLists.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: a regenerated `libwhisper/build/libwhisper.xcodeproj` containing a `whisper.coreml` static-lib target and a `-DWHISPER_USE_COREML` compile flag on the `whisper` target. Determines the exact name/location of the CoreML static archive (expected `libwhisper/build/src/<config>/libwhisper.coreml.a`) that Task 2 must link.

- [ ] **Step 1: Add the CoreML options to the libwhisper CMake config**

In `libwhisper/CMakeLists.txt`, immediately after the `add_compile_options(-U__ARM_FEATURE_MATMUL_INT8)` line and before `add_subdirectory(whisper.cpp)`, add:

```cmake
# Run the Whisper encoder on the Apple Neural Engine via CoreML.
# ALLOW_FALLBACK keeps CPU as a safety net when no -encoder.mlmodelc is present
# or the device/model is incompatible.
set(WHISPER_COREML ON CACHE BOOL "" FORCE)
set(WHISPER_COREML_ALLOW_FALLBACK ON CACHE BOOL "" FORCE)
```

- [ ] **Step 2: Regenerate the Xcode project**

Run: `cmake -G Xcode -B libwhisper/build -S libwhisper`
Expected: configuration succeeds and the log contains `CoreML framework found`.

- [ ] **Step 3: Discover the CoreML target and archive name**

Run: `xcodebuild -project libwhisper/build/libwhisper.xcodeproj -list && find libwhisper/build -name "libwhisper.coreml.a" -o -name "*coreml*.a" 2>/dev/null`
Expected: a target named `whisper.coreml` appears in the target list. (The `.a` may not exist until built — that is fine; Step 4 builds it.)

- [ ] **Step 4: Build libwhisper alone to produce both archives**

Run: `xcodebuild -project libwhisper/build/libwhisper.xcodeproj -target whisper -configuration Debug -destination 'platform=macOS,arch=arm64' build && find libwhisper/build -name "*.a"`
Expected: BUILD SUCCEEDED, and the `find` lists both `libwhisper.a` (from target `whisper`) and `libwhisper.coreml.a` (from target `whisper.coreml`). **Record both absolute paths** — Task 2 needs them. If only `libwhisper.a` exists, open `libwhisper/build/src/CMakeLists`-derived target and confirm `whisper.coreml` is a dependency; the archive name is whatever the target emits.

- [ ] **Step 5: Commit**

```bash
git add libwhisper/CMakeLists.txt
git commit -m "build: enable WHISPER_COREML with CPU fallback in libwhisper"
```

---

## Task 2: Link the CoreML object code and frameworks into the app

**Files:**
- Modify: `OpenSuperWhisper.xcodeproj/project.pbxproj`

**Interfaces:**
- Consumes: `libwhisper.coreml.a` path from Task 1; the app already links `libwhisper.a`, `Accelerate.framework`, `Metal.framework`, `MetalKit.framework` (`project.pbxproj:219-224`).
- Produces: an app binary that contains the CoreML encoder symbols and links `CoreML.framework` + `Foundation.framework`, still building and running with CPU fallback (no `.mlmodelc` downloaded yet).

- [ ] **Step 1: Add the CoreML static archive and frameworks in Xcode**

Open `OpenSuperWhisper.xcodeproj` in Xcode. Select the `OpenSuperWhisper` target → **Build Phases** → **Link Binary With Libraries** → **+**:
1. Add `CoreML.framework` (System framework).
2. Add `Foundation.framework` (System framework) if not already present.
3. Add the `libwhisper.coreml.a` produced in Task 1 (Add Other… → navigate to the path recorded in Task 1 Step 4). Reference it the same way `libwhisper.a` is referenced (relative path under `libwhisper/build`).

Also add `whisper.coreml` as a **Target Dependency** (Build Phases → Dependencies → +) so it builds before the app, mirroring the existing `whisper` dependency (`project.pbxproj:59-97`).

- [ ] **Step 2: Build the full app via the dev script**

Run: `./run.sh build`
Expected: `Building successful!`. The build must not report undefined symbols for `whisper_coreml_*`. If the linker reports duplicate or missing CoreML symbols, confirm only ONE of `libwhisper.a`/`libwhisper.coreml.a` defines them and that both archives are from the same Task 1 build.

- [ ] **Step 3: Launch and confirm CPU fallback still transcribes**

Run: `./run.sh` then record a short phrase with the default `ggml-tiny.en` model (no encoder downloaded yet).
Expected: transcription still works. In the console log, whisper.cpp prints a CoreML load attempt and, because no `.mlmodelc` exists, falls back to CPU (look for a `whisper_init_state: failed to load Core ML model` or equivalent line followed by normal decoding). No crash, output identical to before.

- [ ] **Step 4: Commit**

```bash
git add OpenSuperWhisper.xcodeproj/project.pbxproj
git commit -m "build: link CoreML.framework and whisper.coreml into the app target"
```

---

## Task 3: Validate real ANE acceleration (manual mlmodelc spike — gate)

**Files:** none (verification-only gate before building download/UX code).

**Interfaces:**
- Consumes: a working CoreML-enabled build (Tasks 1-2) and a real Whisper `.bin` already downloaded (e.g. `ggml-large-v3-turbo-q5_0.bin`).
- Produces: a measured before/after latency confirming CoreML is actually used. If no speedup is observed, STOP and reassess before investing in Tasks 4-6.

- [ ] **Step 1: Download a base.en bin + its encoder manually**

```bash
DIR="$HOME/Library/Application Support/com.superboring.opensuperwhisper/whisper-models"
cd "$DIR"
curl -L -o ggml-base.en.bin "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin?download=true"
curl -L -o enc.zip "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en-encoder.mlmodelc.zip?download=true"
unzip -o enc.zip && rm enc.zip
ls -d ggml-base.en-encoder.mlmodelc
```
Expected: `ggml-base.en-encoder.mlmodelc` directory sits next to `ggml-base.en.bin`.

- [ ] **Step 2: Select base.en and transcribe a ~30s clip, watching the log**

Run the app (`./run.sh`), select the `base.en` model, transcribe a ~30 second recording.
Expected log line: `whisper_init_state: Core ML model loaded` (whisper.cpp prints this on success, `whisper.cpp/src/whisper.cpp:3457-3459`). First run is slow (CoreML compiles the model); the second run is fast.

- [ ] **Step 3: Measure speedup**

Transcribe the same clip twice with the encoder present (warm) and once after deleting `ggml-base.en-encoder.mlmodelc` (CPU). Compare the `encode` timing in the whisper.cpp timing summary, or wall-clock the transcription.
Expected: warm CoreML encode is meaningfully faster than CPU (typically 2-3× on the encoder stage). Record the numbers in the commit message of Task 6.

- [ ] **Step 4: No commit (verification gate).** If acceleration is confirmed, proceed to Task 4. If not, stop and reassess.

---

## Task 4: CoreML model-path helper (pure, unit-tested)

**Files:**
- Create: `OpenSuperWhisper/CoreMLModel.swift`
- Test: `OpenSuperWhisperTests/CoreMLModelTests.swift`

**Interfaces:**
- Consumes: nothing (pure string logic).
- Produces:
  - `enum CoreMLModel` with `static func encoderBundleName(forModelFilename: String) -> String` — maps `"ggml-large-v3-turbo-q5_0.bin"` → `"ggml-large-v3-turbo-q5_0-encoder.mlmodelc"` (replace trailing `.bin` with `-encoder.mlmodelc`), matching whisper.cpp's contract.
  - `static func upstreamEncoderZipName(forModelFilename: String) -> String?` — strips a trailing quant suffix (`-q5_0`, `-q8_0`, `-q4_0`, `-q5_1`, `-q4_1`, `-q8_1`) before `.bin` so all turbo quants resolve to the single upstream asset `"ggml-large-v3-turbo-encoder.mlmodelc.zip"`; returns `nil` for models with no known upstream encoder.

- [ ] **Step 1: Write the failing tests**

```swift
import XCTest
@testable import OpenSuperWhisper

final class CoreMLModelTests: XCTestCase {
    func testEncoderBundleNameForPlainModel() {
        XCTAssertEqual(
            CoreMLModel.encoderBundleName(forModelFilename: "ggml-large-v3-turbo.bin"),
            "ggml-large-v3-turbo-encoder.mlmodelc")
    }

    func testEncoderBundleNameForQuantizedModel() {
        XCTAssertEqual(
            CoreMLModel.encoderBundleName(forModelFilename: "ggml-large-v3-turbo-q5_0.bin"),
            "ggml-large-v3-turbo-q5_0-encoder.mlmodelc")
    }

    func testUpstreamZipStripsQuantSuffix() {
        XCTAssertEqual(
            CoreMLModel.upstreamEncoderZipName(forModelFilename: "ggml-large-v3-turbo-q8_0.bin"),
            "ggml-large-v3-turbo-encoder.mlmodelc.zip")
        XCTAssertEqual(
            CoreMLModel.upstreamEncoderZipName(forModelFilename: "ggml-large-v3-turbo.bin"),
            "ggml-large-v3-turbo-encoder.mlmodelc.zip")
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `xcodebuild test -scheme OpenSuperWhisper -destination 'platform=macOS,arch=arm64' -only-testing:OpenSuperWhisperTests/CoreMLModelTests -clonedSourcePackagesDirPath SourcePackages -skipPackagePluginValidation 2>&1 | tail -20`
Expected: FAIL — `CoreMLModel` is undefined / does not compile.

- [ ] **Step 3: Implement the helper**

```swift
import Foundation

enum CoreMLModel {
    private static let quantSuffixes = ["-q5_0", "-q8_0", "-q4_0", "-q5_1", "-q4_1", "-q8_1"]

    /// whisper.cpp expects the encoder bundle as the model path with `.bin`
    /// replaced by `-encoder.mlmodelc` (see whisper_get_coreml_path_encoder).
    static func encoderBundleName(forModelFilename filename: String) -> String {
        let base = filename.hasSuffix(".bin") ? String(filename.dropLast(4)) : filename
        return base + "-encoder.mlmodelc"
    }

    /// Upstream (HuggingFace ggerganov/whisper.cpp) ships one encoder per model
    /// family, shared across quantizations. Strip the quant suffix to find it.
    /// Returns nil for models with no known upstream CoreML encoder.
    static func upstreamEncoderZipName(forModelFilename filename: String) -> String? {
        guard filename.hasSuffix(".bin") else { return nil }
        var base = String(filename.dropLast(4))
        for suffix in quantSuffixes where base.hasSuffix(suffix) {
            base = String(base.dropLast(suffix.count))
            break
        }
        return base + "-encoder.mlmodelc.zip"
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `xcodebuild test -scheme OpenSuperWhisper -destination 'platform=macOS,arch=arm64' -only-testing:OpenSuperWhisperTests/CoreMLModelTests -clonedSourcePackagesDirPath SourcePackages -skipPackagePluginValidation 2>&1 | tail -20`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add OpenSuperWhisper/CoreMLModel.swift OpenSuperWhisperTests/CoreMLModelTests.swift
git commit -m "feat: CoreML encoder path/URL mapping helper"
```

---

## Task 5: Download, unzip, and place the CoreML encoder

**Files:**
- Modify: `OpenSuperWhisper/WhisperModelManager.swift`

**Interfaces:**
- Consumes: `CoreMLModel.encoderBundleName(...)` (Task 4); existing `modelsDirectory`, `downloadModel(url:name:progressCallback:)` patterns.
- Produces:
  - `func isCoreMLEncoderPresent(forModelFilename: String) -> Bool`
  - `func downloadCoreMLEncoder(zipURL: URL, forModelFilename: String, progressCallback: @escaping (Double) -> Void) async throws` — downloads the zip to a temp file, unzips it, locates the `.mlmodelc` directory inside, and moves/renames it to `modelsDirectory/<encoderBundleName>`. Idempotent: returns early if already present.

- [ ] **Step 1: Add encoder presence + download to `WhisperModelManager`**

Append these methods inside `class WhisperModelManager` (after `isModelDownloaded`):

```swift
func isCoreMLEncoderPresent(forModelFilename filename: String) -> Bool {
    let name = CoreMLModel.encoderBundleName(forModelFilename: filename)
    return FileManager.default.fileExists(atPath: modelsDirectory.appendingPathComponent(name).path)
}

/// Downloads and installs the CoreML encoder bundle next to its `.bin`, named
/// exactly as whisper.cpp expects. Idempotent; failures here are non-fatal to
/// transcription (CPU fallback still works) — callers should not treat a thrown
/// error as a model-download failure.
func downloadCoreMLEncoder(zipURL: URL,
                           forModelFilename filename: String,
                           progressCallback: @escaping (Double) -> Void) async throws {
    let targetName = CoreMLModel.encoderBundleName(forModelFilename: filename)
    let targetURL = modelsDirectory.appendingPathComponent(targetName)
    if FileManager.default.fileExists(atPath: targetURL.path) {
        await MainActor.run { progressCallback(1.0) }
        return
    }

    // Reuse the existing download machinery to fetch the zip into modelsDirectory.
    let zipName = targetName + ".download.zip"
    try await downloadModel(url: zipURL, name: zipName, progressCallback: progressCallback)
    let zipURLOnDisk = modelsDirectory.appendingPathComponent(zipName)
    defer { try? FileManager.default.removeItem(at: zipURLOnDisk) }

    // Unzip into a temp dir, then move the inner .mlmodelc to the target path.
    let tmpDir = modelsDirectory.appendingPathComponent(targetName + ".unzip-tmp")
    try? FileManager.default.removeItem(at: tmpDir)
    try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tmpDir) }

    try unzipItem(at: zipURLOnDisk, to: tmpDir)

    guard let inner = try FileManager.default
        .contentsOfDirectory(at: tmpDir, includingPropertiesForKeys: nil)
        .first(where: { $0.pathExtension == "mlmodelc" }) else {
        throw NSError(domain: "WhisperModelManager", code: -2,
                      userInfo: [NSLocalizedDescriptionKey: "No .mlmodelc found in encoder archive"])
    }
    if FileManager.default.fileExists(atPath: targetURL.path) {
        try FileManager.default.removeItem(at: targetURL)
    }
    try FileManager.default.moveItem(at: inner, to: targetURL)
    await MainActor.run { progressCallback(1.0) }
}

/// Unzips using the system `ditto` tool (handles .mlmodelc bundles reliably).
private func unzipItem(at zip: URL, to destination: URL) throws {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
    process.arguments = ["-x", "-k", zip.path, destination.path]
    try process.run()
    process.waitUntilExit()
    if process.terminationStatus != 0 {
        throw NSError(domain: "WhisperModelManager", code: -3,
                      userInfo: [NSLocalizedDescriptionKey: "Failed to unzip encoder archive"])
    }
}
```

- [ ] **Step 2: Build to verify it compiles**

Run: `./run.sh build`
Expected: `Building successful!`.

- [ ] **Step 3: Manual functional check**

Temporarily call from a debug action (or reuse Task 3's bin): delete any existing `ggml-base.en-encoder.mlmodelc`, then exercise `downloadCoreMLEncoder(zipURL: <base.en zip>, forModelFilename: "ggml-base.en.bin")`.
Expected: `ggml-base.en-encoder.mlmodelc` appears in the models dir; the temp zip and unzip-tmp dir are gone; transcription logs `Core ML model loaded`.

- [ ] **Step 4: Commit**

```bash
git add OpenSuperWhisper/WhisperModelManager.swift
git commit -m "feat: download, unzip, and install CoreML encoder bundles"
```

---

## Task 6: Wire encoder download into the model-download flows + UX

**Files:**
- Modify: `OpenSuperWhisper/Settings.swift`
- Modify: `OpenSuperWhisper/Onboarding/OnboardingView.swift`

**Interfaces:**
- Consumes: `WhisperModelManager.downloadCoreMLEncoder(...)`, `isCoreMLEncoderPresent(...)` (Task 5); `CoreMLModel.upstreamEncoderZipName(...)` (Task 4); existing `SettingsDownloadableModel` (`Settings.swift:673-706`) and `downloadModel(_:)` (`Settings.swift:453`).
- Produces: after a Whisper model finishes downloading, the matching CoreML encoder is fetched automatically (best-effort, non-fatal); the Models UI shows whether the Neural Engine encoder is installed.

- [ ] **Step 1: Add the encoder URL to `SettingsDownloadableModel`**

In the `struct SettingsDownloadableModel` (`Settings.swift:673`), add a stored property and init parameter:

```swift
    /// Upstream CoreML encoder archive for this model, if one exists. Downloaded
    /// after the .bin to enable Apple Neural Engine acceleration.
    let coreMLEncoderURL: URL?
```
Add `coreMLEncoderURL: URL? = nil` to the `init(...)` signature (after `preferredLanguage`) and `self.coreMLEncoderURL = coreMLEncoderURL` in the body.

- [ ] **Step 2: Populate it for the three turbo entries**

In `SettingsDownloadableModels.availableModels` (`Settings.swift:709-740`), add to each of the three `large-v3-turbo` entries (NOT the Hebrew/ivrit entry, which has no upstream encoder):

```swift
            coreMLEncoderURL: URL(string: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-encoder.mlmodelc.zip?download=true")
```

- [ ] **Step 3: Trigger encoder download after the .bin in Settings**

In `downloadModel(_ model: SettingsDownloadableModel)` (`Settings.swift:453`), after the existing `try await WhisperModelManager.shared.downloadModel(...)` call succeeds, add:

```swift
        if let encoderURL = model.coreMLEncoderURL {
            // Best-effort: ANE acceleration is a bonus; CPU fallback always works.
            try? await WhisperModelManager.shared.downloadCoreMLEncoder(
                zipURL: encoderURL,
                forModelFilename: filename) { _ in }
        }
```

- [ ] **Step 4: Mirror the trigger in onboarding**

In `OnboardingView.swift` `downloadModel(_:)` (`OnboardingView.swift:115-140`), after the `.bin` download completes, add the same best-effort `downloadCoreMLEncoder` call using that flow's `url`/`filename` and the corresponding encoder URL (map via `CoreMLModel.upstreamEncoderZipName(forModelFilename: filename)` joined to the HuggingFace base `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/` + `?download=true`).

- [ ] **Step 5: Add a "Neural Engine" status row in the Models UI**

In the model row view in `Settings.swift` (near where `isDownloaded` is shown, around `Settings.swift:2217-2454`), when a model is downloaded show a small caption:

```swift
                if WhisperModelManager.shared.isCoreMLEncoderPresent(forModelFilename: model.filename) {
                    Label("Neural Engine ready", systemImage: "bolt.fill")
                        .font(.caption).foregroundStyle(.green)
                } else if model.coreMLEncoderURL != nil {
                    Text("First transcription compiles the Neural Engine model (one-time, a few seconds).")
                        .font(.caption).foregroundStyle(.secondary)
                }
```

- [ ] **Step 6: Build and run end-to-end**

Run: `./run.sh`
Then: in Settings → Models, download `Turbo V3 small` (q5_0). Confirm both `ggml-large-v3-turbo-q5_0.bin` and `ggml-large-v3-turbo-q5_0-encoder.mlmodelc` land in the models dir, the "Neural Engine ready" label appears, and the first transcription logs `Core ML model loaded`.
Expected: faster transcription on the second run; CPU fallback if the encoder is deleted.

- [ ] **Step 7: Commit**

```bash
git add OpenSuperWhisper/Settings.swift OpenSuperWhisper/Onboarding/OnboardingView.swift
git commit -m "feat: auto-download CoreML encoders and show Neural Engine status"
```

---

## Task 7: Release/build-flow + docs

**Files:**
- Modify: `make_release.sh`, `notarize_app.sh` (verify only), `README.md`

**Interfaces:**
- Consumes: the working feature (Tasks 1-6).
- Produces: release builds that include CoreML linkage on arm64 and still build on x86_64 (CPU-only), plus user-facing docs.

- [ ] **Step 1: Confirm the release build enables CoreML on arm64 and still builds x86_64**

Run: `./make_release.sh` (or its arm64 path) and confirm the arm64 build links CoreML and the x86_64 build still succeeds (CoreML target is Apple-Silicon-only; on Intel the encoder simply never loads → CPU path).
Expected: both arch builds succeed; notarization step unaffected.

- [ ] **Step 2: Document the feature**

In `README.md`, under the engines/requirements section, add a short note: Apple Neural Engine acceleration for Whisper is automatic on Apple Silicon when a model is downloaded (its CoreML encoder is fetched alongside); the first transcription after install compiles the encoder (one-time, a few seconds); Intel Macs use the CPU path.

- [ ] **Step 3: Commit**

```bash
git add README.md make_release.sh
git commit -m "docs: document Whisper Neural Engine (CoreML) acceleration"
```

---

## Self-Review Notes

- **Spec coverage:** build enablement (T1), linkage (T2), real-accel verification gate (T3), path helper (T4), download/unzip/rename (T5), wiring + UX (T6), release + docs (T7). All chosen-scope items (build + download + UX) covered. `#1` relabeling intentionally excluded per decision.
- **Fallback constraint:** enforced in T1 (`ALLOW_FALLBACK`), verified in T2 Step 3 and T6 Step 6.
- **Path contract:** `CoreMLModel.encoderBundleName` (T4) matches whisper.cpp `whisper_get_coreml_path_encoder`; quant variants handled by `upstreamEncoderZipName`.
- **Type consistency:** `encoderBundleName(forModelFilename:)`, `upstreamEncoderZipName(forModelFilename:)`, `downloadCoreMLEncoder(zipURL:forModelFilename:progressCallback:)`, `isCoreMLEncoderPresent(forModelFilename:)` used consistently across T4-T6.
- **Open risk:** T1 Step 4 / T2 Step 2 — whether `libwhisper.coreml.a` must be linked separately or is folded into `libwhisper.a`. T1 records the actual archive(s); T2 links whatever T1 produced. This is the one genuine build unknown and is deliberately the first thing validated.
