//
//  ContentView.swift
//  Coding Journal 2 - Core ML
//
//  Created by Michelle Thomas on 4/21/25.
//

import SwiftUI
import CoreML
import Vision

// Define Line struct
struct Line: Identifiable {
    let id = UUID()
    var points: [CGPoint]
}

struct ContentView: View {
    @State private var lines: [Line] = []
    @State private var currentLine: Line?
    @State private var predictionText: String = "Draw something!"
    @State private var showingTeachSheet: Bool = false
    @State private var showingResetAlert: Bool = false
    @State private var showingStatsView: Bool = false
    
    // Use ObservedObject to automatically update when stats change
    @ObservedObject var classifier = DrawingClassifier()
    
    var body: some View {
        VStack {
            Text("DrawPedia")
                .font(.largeTitle)
                .fontWeight(.bold)
                .padding()
            
            // Drawing canvas
            ZStack {
                Rectangle()
                    .fill(Color.white)
                    .border(Color.gray, width: 1)
                
                Canvas { context, size in
                    for line in lines {
                        var path = Path()
                        path.addLines(line.points)
                        context.stroke(path, with: .color(.black), lineWidth: 3)
                    }
                    
                    if let line = currentLine {
                        var path = Path()
                        path.addLines(line.points)
                        context.stroke(path, with: .color(.black), lineWidth: 3)
                    }
                }
            }
            .frame(height: 300)
            .background(Color.white)
            .gesture(
                DragGesture(minimumDistance: 0, coordinateSpace: .local)
                    .onChanged { value in
                        let location = value.location
                        if currentLine == nil {
                            currentLine = Line(points: [location])
                        } else {
                            currentLine?.points.append(location)
                        }
                    }
                    .onEnded { value in
                        if let line = currentLine {
                            lines.append(line)
                            currentLine = nil
                            // Enable classification
                            classifyDrawing()
                        }
                    }
            )
            
            Text(predictionText)
                .font(.title2)
                .padding()
            
            HStack(spacing: 20) {
                Button("Clear") {
                    lines = []
                    currentLine = nil
                    predictionText = "Draw something!"
                }
                .buttonStyle(.bordered)
                
                Button("Teach") {
                    showingTeachSheet = true
                }
                .buttonStyle(.borderedProminent)
                
                Button("Stats") {
                    showingStatsView = true
                }
                .buttonStyle(.bordered)
                
                Button("Reset") {
                    showingResetAlert = true
                }
                .buttonStyle(.bordered)
                .foregroundColor(.red)
            }
            .padding()
        }
        .padding()
        .background(Color(.systemBackground))
        .sheet(isPresented: $showingTeachSheet) {
            TeachView { label in
                teachModel(label: label)
            }
        }
        .sheet(isPresented: $showingStatsView) {
            StatsView(stats: classifier.modelStats)
        }
        .alert("Reset Model", isPresented: $showingResetAlert) {
            Button("Cancel", role: .cancel) { }
            Button("Reset", role: .destructive) {
                resetModel()
            }
        } message: {
            Text("Are you sure you want to reset the model? This will erase all training data.")
        }
    }
    
    // Add classification function
    func classifyDrawing() {
        guard !lines.isEmpty else {
            predictionText = "Draw something!"
            return
        }
        
        predictionText = "Analyzing..."
        
        // Create a UIImage from our drawing
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: 299, height: 299))
        let image = renderer.image { ctx in
            // Fill with white background
            UIColor.white.setFill()
            ctx.fill(CGRect(x: 0, y: 0, width: 299, height: 299))
            
            // Draw all lines in black
            UIColor.black.setStroke()
            
            for line in lines {
                guard let firstPoint = line.points.first else { continue }
                
                let path = UIBezierPath()
                path.lineWidth = 3.0
                path.lineCapStyle = .round
                path.lineJoinStyle = .round
                
                path.move(to: firstPoint)
                for point in line.points.dropFirst() {
                    path.addLine(to: point)
                }
                
                path.stroke()
            }
        }
        
        // Now use this image for classification
        classifier.classifyDrawingFromImage(image) { label, confidence in
            if let label = label, let confidence = confidence, confidence > 0.3 {
                DispatchQueue.main.async {
                    self.predictionText = "I see a \(label) (Confidence: \(Int(confidence * 100))%)"
                }
            } else {
                DispatchQueue.main.async {
                    self.predictionText = "I'm not sure what that is. Can you teach me?"
                }
            }
        }
    }
    
    // Add drawing view for rendering
    var drawingView: some View {
        Canvas { context, size in
            for line in lines {
                var path = Path()
                path.addLines(line.points)
                context.stroke(path, with: .color(.black), lineWidth: 3)
            }
            
            if let line = currentLine {
                var path = Path()
                path.addLines(line.points)
                context.stroke(path, with: .color(.black), lineWidth: 3)
            }
        }
        .frame(width: 299, height: 299)  // Standard size for ML model input
        .background(Color.white)
    }
    
    // Add teaching function
    func teachModel(label: String) {
        guard !lines.isEmpty, !label.isEmpty else { return }
        
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: 299, height: 299))
        let image = renderer.image { ctx in
            // Fill with white background
            UIColor.white.setFill()
            ctx.fill(CGRect(x: 0, y: 0, width: 299, height: 299))
            
            // Draw all lines in black
            UIColor.black.setStroke()
            
            for line in lines {
                guard let firstPoint = line.points.first else { continue }
                
                let path = UIBezierPath()
                path.lineWidth = 3.0
                path.lineCapStyle = .round
                path.lineJoinStyle = .round
                
                path.move(to: firstPoint)
                for point in line.points.dropFirst() {
                    path.addLine(to: point)
                }
                
                path.stroke()
            }
        }
        
        classifier.addTrainingExampleFromImage(image, label: label)
        predictionText = "Taught as: \(label)"
    }
    
    // Add reset function
    func resetModel() {
        classifier.resetModel()
        predictionText = "Model reset. Draw something!"
    }
    
    // If using the standard approach, add this to update stats:
    func updateStats() {
        // Assuming you have a modelStats property in your ContentView
        // This function should update the modelStats property with the latest stats from the classifier
        // You might want to implement this based on your specific requirements
    }
    
    // Improved rendering function with normalization
    func renderDrawingToImage() -> UIImage {
        // Get the bounding box of the drawing
        var minX: CGFloat = .infinity
        var minY: CGFloat = .infinity
        var maxX: CGFloat = 0
        var maxY: CGFloat = 0
        
        for line in lines {
            for point in line.points {
                minX = min(minX, point.x)
                minY = min(minY, point.y)
                maxX = max(maxX, point.x)
                maxY = max(maxY, point.y)
            }
        }
        
        // Ensure we have a valid bounding box
        if minX == .infinity || lines.isEmpty {
            // Return blank image if no drawing
            let renderer = UIGraphicsImageRenderer(size: CGSize(width: 299, height: 299))
            return renderer.image { ctx in
                UIColor.white.setFill()
                ctx.fill(CGRect(x: 0, y: 0, width: 299, height: 299))
            }
        }
        
        // Add padding
        let padding: CGFloat = 20
        minX = max(0, minX - padding)
        minY = max(0, minY - padding)
        maxX = maxX + padding
        maxY = maxY + padding
        
        // Calculate scale to fit drawing in target size while preserving aspect ratio
        let drawingWidth = maxX - minX
        let drawingHeight = maxY - minY
        let targetSize = CGSize(width: 299, height: 299)
        
        let scaleX = targetSize.width / drawingWidth
        let scaleY = targetSize.height / drawingHeight
        let scale = min(scaleX, scaleY) * 0.9 // 90% to add a margin
        
        // Calculate centering offset
        let scaledWidth = drawingWidth * scale
        let scaledHeight = drawingHeight * scale
        let offsetX = (targetSize.width - scaledWidth) / 2
        let offsetY = (targetSize.height - scaledHeight) / 2
        
        // Render normalized drawing
        let renderer = UIGraphicsImageRenderer(size: targetSize)
        return renderer.image { ctx in
            // Fill background
            UIColor.white.setFill()
            ctx.fill(CGRect(origin: .zero, size: targetSize))
            
            // Set up drawing
            let context = ctx.cgContext
            context.setLineCap(.round)
            context.setLineJoin(.round)
            context.setLineWidth(3.0)
            UIColor.black.setStroke()
            
            // Draw lines with normalization
            for line in lines {
                guard let firstPoint = line.points.first else { continue }
                
                // Transform first point
                let normalizedFirstX = (firstPoint.x - minX) * scale + offsetX
                let normalizedFirstY = (firstPoint.y - minY) * scale + offsetY
                
                context.beginPath()
                context.move(to: CGPoint(x: normalizedFirstX, y: normalizedFirstY))
                
                // Transform and draw rest of the line
                for point in line.points.dropFirst() {
                    let normalizedX = (point.x - minX) * scale + offsetX
                    let normalizedY = (point.y - minY) * scale + offsetY
                    context.addLine(to: CGPoint(x: normalizedX, y: normalizedY))
                }
                
                context.strokePath()
            }
        }
    }
}

// Add TeachView
struct TeachView: View {
    let onSubmit: (String) -> Void
    @State private var newLabel: String = ""
    @Environment(\.dismiss) var dismiss
    
    var body: some View {
        NavigationView {
            VStack {
                Text("What did you draw?")
                    .font(.title2)
                    .padding()
                
                TextField("Enter label (e.g., cat, tree)", text: $newLabel)
                    .textFieldStyle(.roundedBorder)
                    .padding()
                
                Button("Submit") {
                    onSubmit(newLabel)
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .disabled(newLabel.isEmpty)
                .padding()
            }
            .padding()
            .navigationTitle("Teach DrawPedia")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }
}

// Add StatsView
struct StatsView: View {
    let stats: [String: Int]
    @Environment(\.dismiss) var dismiss
    
    var body: some View {
        NavigationView {
            List {
                if stats.isEmpty {
                    Text("No training data yet")
                        .foregroundColor(.secondary)
                } else {
                    ForEach(Array(stats.keys.sorted()), id: \.self) { label in
                        HStack {
                            Text(label)
                            Spacer()
                            Text("\(stats[label, default: 0]) examples")
                                .foregroundColor(.secondary)
                        }
                    }
                }
            }
            .navigationTitle("Model Stats")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

#Preview {
    ContentView()
} 