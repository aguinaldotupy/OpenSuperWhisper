import Foundation

/// Maps a Whisper `.bin` model filename to the CoreML encoder bundle that
/// whisper.cpp expects beside it, and to the upstream archive that ships it.
enum CoreMLModel {
    private static let quantSuffixes = ["-q5_0", "-q8_0", "-q4_0", "-q5_1", "-q4_1", "-q8_1"]

    /// Model families that ship a CoreML encoder on HuggingFace
    /// (ggerganov/whisper.cpp). Used to honor the "nil for unsupported" contract
    /// so callers don't attempt downloads that would 404.
    private static let knownEncoderBases: Set<String> = [
        "ggml-tiny", "ggml-tiny.en", "ggml-base", "ggml-base.en",
        "ggml-small", "ggml-small.en", "ggml-medium", "ggml-medium.en",
        "ggml-large-v1", "ggml-large-v2", "ggml-large-v3", "ggml-large-v3-turbo",
    ]

    /// whisper.cpp expects the encoder bundle as the model path with `.bin`
    /// replaced by `-encoder.mlmodelc` (see `whisper_get_coreml_path_encoder`
    /// in whisper.cpp/src/whisper.cpp).
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
        guard knownEncoderBases.contains(base) else { return nil }
        return base + "-encoder.mlmodelc.zip"
    }
}
