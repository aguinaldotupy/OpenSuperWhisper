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

    func testUpstreamZipReturnsNilWhenUnsupported() {
        // Not a .bin
        XCTAssertNil(CoreMLModel.upstreamEncoderZipName(forModelFilename: "notes.txt"))
        // A .bin with no known upstream encoder family
        XCTAssertNil(CoreMLModel.upstreamEncoderZipName(forModelFilename: "custom-model.bin"))
        // The Hebrew ivrit model has no upstream CoreML encoder
        XCTAssertNil(CoreMLModel.upstreamEncoderZipName(forModelFilename: "ggml-ivrit-large-v3-turbo.bin"))
    }
}
