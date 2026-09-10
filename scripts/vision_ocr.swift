import Foundation
import Vision
import AppKit
let url = URL(fileURLWithPath: CommandLine.arguments[1])
let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = false
request.recognitionLanguages = ["en-US"]
let handler = VNImageRequestHandler(url: url, options: [:])
try handler.perform([request])
let blocks: [[String: Any]] = (request.results ?? []).compactMap { observation in
    guard let text = observation.topCandidates(1).first else { return nil }
    let box = observation.boundingBox
    return ["type": "line", "text": text.string, "confidence": text.confidence,
            "bbox": [box.minX, 1-box.maxY, box.maxX, 1-box.minY]]
}
let data = try JSONSerialization.data(withJSONObject: blocks, options: [.prettyPrinted, .sortedKeys])
FileHandle.standardOutput.write(data)
