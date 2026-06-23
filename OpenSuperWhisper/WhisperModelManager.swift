import Combine
import Foundation

class WhisperDownloadDelegate: NSObject, URLSessionTaskDelegate, URLSessionDownloadDelegate {
    private let progressCallback: (Double) -> Void
    private var expectedContentLength: Int64 = 0
    var completionHandler: ((URL?, Error?) -> Void)?
    weak var downloadTask: URLSessionDownloadTask?
    
    init(progressCallback: @escaping (Double) -> Void) {
        self.progressCallback = progressCallback
        super.init()
    }
    
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        completionHandler?(location, nil)
    }
    
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
      
        if expectedContentLength == 0 {
            expectedContentLength = totalBytesExpectedToWrite
        }
        let progress = Double(totalBytesWritten) / Double(expectedContentLength)
        
        DispatchQueue.main.async { [weak self] in
            self?.progressCallback(progress)
        }

    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didResumeAtOffset fileOffset: Int64, expectedTotalBytes: Int64) {
    }
    
    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error = error {
            completionHandler?(nil, error)
        } else {
        }
    }
}

class WhisperModelManager {
    static let shared = WhisperModelManager()
    
    private let modelsDirectoryName = "whisper-models"
    private var activeDownloadTasks: [String: URLSessionDownloadTask] = [:]
    private let downloadTasksLock = NSLock()
    
    var modelsDirectory: URL {
        let applicationSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let modelsDirectory = applicationSupport.appendingPathComponent(Bundle.main.bundleIdentifier!).appendingPathComponent(modelsDirectoryName)
        return modelsDirectory
    }
    
    private init() {
        createModelsDirectoryIfNeeded()
        copyDefaultModelIfNeeded()
    }
    
    private func createModelsDirectoryIfNeeded() {
        do {
            try FileManager.default.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
        } catch {
            print("Failed to create models directory: \(error)")
        }
    }
    
    private func copyDefaultModelIfNeeded() {
        let defaultModelName = "ggml-tiny.en.bin"
        let destinationURL = modelsDirectory.appendingPathComponent(defaultModelName)
        
        // Check if model already exists
        if FileManager.default.fileExists(atPath: destinationURL.path) {
            return
        }
        
        // Look for the model in the bundle
        if let bundleURL = Bundle.main.url(forResource: "ggml-tiny.en", withExtension: "bin") {
            do {
                try FileManager.default.copyItem(at: bundleURL, to: destinationURL)
                print("Copied default model to: \(destinationURL.path)")
            } catch {
                print("Failed to copy default model: \(error)")
            }
        }
    }

    // Call this on every startup to ensure at least one model is present
    public func ensureDefaultModelPresent() {
        let defaultModelName = "ggml-tiny.en.bin"
        let destinationURL = modelsDirectory.appendingPathComponent(defaultModelName)
        if !FileManager.default.fileExists(atPath: destinationURL.path) {
            copyDefaultModelIfNeeded()
        }
    }
    
    func getAvailableModels() -> [URL] {
        do {
            return try FileManager.default.contentsOfDirectory(at: modelsDirectory, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension == "bin" }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
        } catch {
            print("Failed to get available models: \(error)")
            return []
        }
    }
    
    // Download model with progress callback using delegate
    func downloadModel(url: URL, name: String, progressCallback: @escaping (Double) -> Void) async throws {
        let destinationURL = modelsDirectory.appendingPathComponent(name)
        
        // Check if model already exists
        if FileManager.default.fileExists(atPath: destinationURL.path) {
            print("Model already exists at: \(destinationURL.path)")
            DispatchQueue.main.async {
                progressCallback(1.0)
            }
            return
        }
        
        print("Starting model download:")
        print("- URL: \(url.absoluteString)")
        print("- Destination: \(destinationURL.path)")
        
        return try await withCheckedThrowingContinuation { continuation in
            let delegate = WhisperDownloadDelegate(progressCallback: progressCallback)
            let configuration = URLSessionConfiguration.default
            configuration.waitsForConnectivity = true
            configuration.timeoutIntervalForResource = 600 // 10 minutes timeout for large models
            
            let session = URLSession(configuration: configuration, delegate: delegate, delegateQueue: .main)
            print("Initiating download...")
            
            // Create a download task without completion handler
            let downloadTask = session.downloadTask(with: url)
            delegate.downloadTask = downloadTask
            
            // Store task for cancellation
            downloadTasksLock.lock()
            activeDownloadTasks[name] = downloadTask
            downloadTasksLock.unlock()
            
            // Add completion handling to delegate
            delegate.completionHandler = { [weak self] location, error in
                // Remove task from active downloads
                self?.downloadTasksLock.lock()
                self?.activeDownloadTasks.removeValue(forKey: name)
                self?.downloadTasksLock.unlock()
                
                // Check if cancelled
                if let error = error as? URLError, error.code == .cancelled {
                    print("Download cancelled")
                    continuation.resume(throwing: CancellationError())
                    return
                }
                
                if let error = error {
                    print("Download failed with error: \(error)")
                    continuation.resume(throwing: error)
                    return
                }
                
                guard let location = location else {
                    let error = NSError(domain: "WhisperModelManager", code: -1, userInfo: [NSLocalizedDescriptionKey: "No download URL received"])
                    continuation.resume(throwing: error)
                    return
                }
                
                do {
                    print("Download completed. Moving file to destination...")
                    try FileManager.default.moveItem(at: location, to: destinationURL)
                    print("Model successfully saved to: \(destinationURL.path)")
                    
                    DispatchQueue.main.async {
                        progressCallback(1.0)
                    }
                    
                    continuation.resume(returning: ())
                } catch {
                    print("Failed to move downloaded file: \(error)")
                    continuation.resume(throwing: error)
                }
            }
            
            downloadTask.resume()
        }
    }
    
    // Cancel download task
    func cancelDownload(name: String) {
        downloadTasksLock.lock()
        defer { downloadTasksLock.unlock() }
        
        if let task = activeDownloadTasks[name] {
            task.cancel()
            activeDownloadTasks.removeValue(forKey: name)
            print("Cancelled download for: \(name)")
        }
    }
    
    // Check if specific model is downloaded
    func isModelDownloaded(name: String) -> Bool {
        let modelPath = modelsDirectory.appendingPathComponent(name).path
        return FileManager.default.fileExists(atPath: modelPath)
    }

    // MARK: - CoreML encoder (Apple Neural Engine acceleration)

    /// Whether the CoreML encoder bundle for a given model `.bin` is installed
    /// next to it (the path whisper.cpp auto-loads to run the encoder on the ANE).
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

        // Find the .mlmodelc bundle anywhere under tmpDir — some archives wrap it
        // in a nested folder. The enumerator yields the bundle directory before
        // its contents, so the first match is the bundle itself.
        var inner: URL?
        if let enumerator = FileManager.default.enumerator(at: tmpDir, includingPropertiesForKeys: nil) {
            for case let url as URL in enumerator where url.pathExtension == "mlmodelc" {
                inner = url
                break
            }
        }
        guard let inner else {
            throw NSError(domain: "WhisperModelManager", code: -2,
                          userInfo: [NSLocalizedDescriptionKey: "No .mlmodelc found in encoder archive"])
        }
        if FileManager.default.fileExists(atPath: targetURL.path) {
            try FileManager.default.removeItem(at: targetURL)
        }
        try FileManager.default.moveItem(at: inner, to: targetURL)
        await MainActor.run { progressCallback(1.0) }
    }

    /// Unzips using the system `ditto` tool (handles `.mlmodelc` bundles reliably).
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
}
