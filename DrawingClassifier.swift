import Foundation
import CoreML
import Vision
import UIKit
import Combine

class DrawingClassifier: ObservableObject {
    @Published private(set) var modelStats: [String: Int] = [:]
    private var model: MLModel?
    private var modelURL: URL?
    private var trainingData: [String: [UIImage]] = [:]
    
    init() {
        loadModel()
    }
    
    private func loadModel() {
        // Try to load the model from the app bundle
        if let bundleModelURL = Bundle.main.url(forResource: "UpdatableDrawingClassifier", withExtension: "mlmodel") {
            do {
                let compiledModelURL = try MLModel.compileModel(at: bundleModelURL)
                self.model = try MLModel(contentsOf: compiledModelURL)
                print("Successfully loaded bundled model")
            } catch {
                print("Error loading bundled model: \(error)")
            }
        } else {
            print("Could not find model in bundle")
        }
        
        // Load any saved training data
        loadTrainingData()
        updateModelStats()
    }
    
    func classifyDrawingFromImage(_ image: UIImage, completion: @escaping (String?, Float?) -> Void) {
        // First try CoreML model if available
        if let model = self.model {
            classifyWithCoreML(image, model: model) { label, confidence in
                if let label = label, let confidence = confidence, confidence > 0.4 {
                    completion(label, confidence)
                } else {
                    // Fall back to our custom classifier
                    self.classifyWithTrainingData(image, completion: completion)
                }
            }
        } else {
            // Use our custom classifier
            classifyWithTrainingData(image, completion: completion)
        }
    }
    
    private func classifyWithCoreML(_ image: UIImage, model: MLModel, completion: @escaping (String?, Float?) -> Void) {
        guard let cgImage = image.cgImage else {
            print("Cannot convert UIImage to CGImage")
            completion(nil, nil)
            return
        }
        
        do {
            let vnModel = try VNCoreMLModel(for: model)
            let request = VNCoreMLRequest(model: vnModel) { request, error in
                if let error = error {
                    print("Classification error: \(error)")
                    completion(nil, nil)
                    return
                }
                
                guard let results = request.results as? [VNClassificationObservation] else {
                    print("Unexpected result type")
                    completion(nil, nil)
                    return
                }
                
                if results.isEmpty {
                    print("No classification results")
                    completion(nil, nil)
                    return
                }
                
                let topResult = results.first!
                print("Top classification: \(topResult.identifier) with confidence \(topResult.confidence)")
                completion(topResult.identifier, topResult.confidence)
            }
            
            // Ensure the image is properly oriented
            request.imageCropAndScaleOption = .centerCrop
            
            let handler = VNImageRequestHandler(cgImage: cgImage)
            try handler.perform([request])
        } catch {
            print("Vision error: \(error)")
            completion(nil, nil)
        }
    }
    
    private func classifyWithTrainingData(_ image: UIImage, completion: @escaping (String?, Float?) -> Void) {
        if trainingData.isEmpty {
            completion(nil, nil)
            return
        }
        
        // Extract features from the drawing
        let drawingFeatures = extractSimpleFeatures(from: image)
        
        // Use k-NN for classification
        let (label, confidence) = classifyWithKNN(drawingFeatures)
        
        completion(label, confidence)
    }
    
    // k-Nearest Neighbors classification
    private func classifyWithKNN(_ features: [Float], k: Int = 3) -> (String?, Float?) {
        if trainingData.isEmpty {
            return (nil, nil)
        }
        
        var allExamples: [(label: String, distance: Float)] = []
        
        // Calculate distance to all training examples
        for (label, images) in trainingData {
            for image in images {
                let exampleFeatures = extractSimpleFeatures(from: image)
                let distance = calculateDistance(features, exampleFeatures)
                allExamples.append((label, distance))
            }
        }
        
        // Sort by distance (smallest first)
        allExamples.sort { $0.distance < $1.distance }
        
        // Take the k nearest examples
        let kNearest = Array(allExamples.prefix(min(k, allExamples.count)))
        
        // Count votes for each label
        var votes: [String: Int] = [:]
        for example in kNearest {
            votes[example.label, default: 0] += 1
        }
        
        // Find the label with the most votes
        if let (bestLabel, voteCount) = votes.max(by: { $0.value < $1.value }) {
            // Calculate confidence based on proportion of votes
            let confidence = Float(voteCount) / Float(kNearest.count)
            return (bestLabel, confidence)
        }
        
        return (nil, nil)
    }
    
    // Calculate Euclidean distance between feature vectors
    private func calculateDistance(_ features1: [Float], _ features2: [Float]) -> Float {
        let minSize = min(features1.count, features2.count)
        var sumSquaredDifferences: Float = 0
        
        for i in 0..<minSize {
            let diff = features1[i] - features2[i]
            sumSquaredDifferences += diff * diff
        }
        
        return sqrt(sumSquaredDifferences)
    }
    
    // Extract a simplified feature set from the image
    private func extractSimpleFeatures(from image: UIImage) -> [Float] {
        let size = 16 // Use a 16x16 grid
        var features: [Float] = []
        
        // Resize the image to our grid size
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1.0
        
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: size, height: size), format: format)
        let resizedImage = renderer.image { context in
            image.draw(in: CGRect(x: 0, y: 0, width: size, height: size))
        }
        
        // Convert to grayscale grid
        if let cgImage = resizedImage.cgImage {
            let width = cgImage.width
            let height = cgImage.height
            let bytesPerPixel = 4
            let bytesPerRow = bytesPerPixel * width
            let bitsPerComponent = 8
            
            var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
            
            let colorSpace = CGColorSpaceCreateDeviceRGB()
            let context = CGContext(data: &pixelData,
                                   width: width,
                                   height: height,
                                   bitsPerComponent: bitsPerComponent,
                                   bytesPerRow: bytesPerRow,
                                   space: colorSpace,
                                   bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            
            context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            
            // Create feature vector from grid
            for y in 0..<height {
                for x in 0..<width {
                    let offset = (y * bytesPerRow) + (x * bytesPerPixel)
                    // Calculate grayscale value
                    let r = pixelData[offset]
                    let g = pixelData[offset + 1]
                    let b = pixelData[offset + 2]
                    let gray = (Float(r) + Float(g) + Float(b)) / (3.0 * 255.0)
                    
                    // Add inverted value (so drawing=1, background=0)
                    features.append(1.0 - gray)
                }
            }
        }
        
        // If we couldn't process the image, return empty features
        if features.isEmpty {
            features = Array(repeating: 0, count: size * size)
        }
        
        return features
    }
    
    func addTrainingExampleFromImage(_ image: UIImage, label: String) {
        // Store the original image
        if trainingData[label] == nil {
            trainingData[label] = []
        }
        trainingData[label]?.append(image)
        
        // Create augmented versions of the image for better training
        let augmentedImages = createAugmentedImages(image)
        trainingData[label]?.append(contentsOf: augmentedImages)
        
        print("Added \(augmentedImages.count + 1) training examples for label '\(label)'")
        
        // Update stats
        updateModelStats()
    }
    
    // Create slightly modified versions of the image for better training
    private func createAugmentedImages(_ image: UIImage) -> [UIImage] {
        var augmentedImages: [UIImage] = []
        
        // 1. Slight rotation variations
        for angle in [-5.0, 5.0] {
            if let rotated = rotateImage(image, byDegrees: angle) {
                augmentedImages.append(rotated)
            }
        }
        
        // 2. Slight scale variations
        for scale in [0.9, 1.1] {
            if let scaled = scaleImage(image, byFactor: scale) {
                augmentedImages.append(scaled)
            }
        }
        
        return augmentedImages
    }
    
    // Rotate image by specified angle
    private func rotateImage(_ image: UIImage, byDegrees degrees: Double) -> UIImage? {
        let radians = degrees * .pi / 180.0
        let rotatedSize = CGRect(origin: .zero, size: image.size)
            .applying(CGAffineTransform(rotationAngle: CGFloat(radians)))
            .size
        
        UIGraphicsBeginImageContextWithOptions(rotatedSize, false, image.scale)
        guard let context = UIGraphicsGetCurrentContext() else {
            return nil
        }
        
        // Move origin to center for rotation
        context.translateBy(x: rotatedSize.width / 2, y: rotatedSize.height / 2)
        context.rotate(by: CGFloat(radians))
        
        // Draw the image centered
        image.draw(in: CGRect(
            x: -image.size.width / 2,
            y: -image.size.height / 2,
            width: image.size.width,
            height: image.size.height))
        
        let rotatedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return rotatedImage
    }
    
    // Scale image by specified factor
    private func scaleImage(_ image: UIImage, byFactor factor: CGFloat) -> UIImage? {
        let newSize = CGSize(
            width: image.size.width * factor,
            height: image.size.height * factor)
        
        UIGraphicsBeginImageContextWithOptions(newSize, false, image.scale)
        image.draw(in: CGRect(origin: .zero, size: newSize))
        let scaledImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return scaledImage
    }
    
    func resetModel() {
        // Clear training data
        trainingData = [:]
        saveTrainingData()
        
        // Reload original model
        loadModel()
        
        // Update stats
        updateModelStats()
    }
    
    // This is the method that was missing
    func getModelStats() -> [String: Int] {
        return modelStats
    }
    
    private func updateModelStats() {
        var stats: [String: Int] = [:]
        
        for (label, images) in trainingData {
            stats[label] = images.count
        }
        
        self.modelStats = stats
    }
    
    // MARK: - Data Persistence
    
    private func saveTrainingData() {
        // This is a simplified version that doesn't actually save images
        // In a real app, you'd save the images to disk and store metadata
        
        let counts = modelStats
        print("Would save training data with counts: \(counts)")
        
        // In a real implementation, you would:
        // 1. Convert images to Data
        // 2. Save them to files
        // 3. Save a metadata file with labels
    }
    
    private func loadTrainingData() {
        // This is a placeholder that would load training data from disk
        // In a real app, you'd read saved images and metadata
        
        print("Would load training data from disk")
        
        // In a real implementation, you would:
        // 1. Read metadata file with labels
        // 2. Load image files
        // 3. Reconstruct the trainingData dictionary
    }
}

// Extension to help with image processing
extension CGContext {
    func colorAt(point: CGPoint) -> UIColor {
        let pixelData = self.data?.advanced(by: Int(point.y) * self.bytesPerRow + Int(point.x) * 4)
        
        if let pixelData = pixelData {
            let r = CGFloat(pixelData.load(as: UInt8.self)) / 255.0
            let g = CGFloat(pixelData.advanced(by: 1).load(as: UInt8.self)) / 255.0
            let b = CGFloat(pixelData.advanced(by: 2).load(as: UInt8.self)) / 255.0
            let a = CGFloat(pixelData.advanced(by: 3).load(as: UInt8.self)) / 255.0
            
            return UIColor(red: r, green: g, blue: b, alpha: a)
        }
        
        return UIColor.black
    }
}

extension UIColor {
    var grayscale: CGFloat {
        var white: CGFloat = 0
        var alpha: CGFloat = 0
        getWhite(&white, alpha: &alpha)
        return white
    }
}

func analyzeDrawingShape(_ image: UIImage, completion: @escaping ([VNRectangleObservation]?, [VNContour]?) -> Void) {
    guard let cgImage = image.cgImage else {
        completion(nil, nil)
        return
    }
    
    // Detect rectangles
    let rectangleRequest = VNDetectRectanglesRequest()
    rectangleRequest.minimumAspectRatio = 0.1
    rectangleRequest.maximumAspectRatio = 10.0
    rectangleRequest.minimumSize = 0.1
    rectangleRequest.maximumObservations = 5
    
    // Detect contours
    let contourRequest = VNDetectContoursRequest()
    contourRequest.contrastAdjustment = 2.0
    contourRequest.detectDarkOnLight = true
    
    // Create request handler
    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    
    do {
        try handler.perform([rectangleRequest, contourRequest])
        
        let rectangles = rectangleRequest.results
        
        // Fixed: VNContoursObservation directly contains contours as a topLevelContours property
        let contours: [VNContour]
        if let contoursObservation = contourRequest.results?.first as? VNContoursObservation {
            contours = contoursObservation.topLevelContours
        } else {
            contours = []
        }
        
        completion(rectangles, contours)
    } catch {
        print("Vision error: \(error)")
        completion(nil, nil)
    }
}
}