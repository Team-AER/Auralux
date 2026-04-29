import Accelerate
import SwiftUI

struct SpectrumAnalyzerView: View {
    var magnitudes: [Float]
    var isPlaying: Bool

    private let barCount = 36

    var body: some View {
        Canvas { context, size in
            let barWidth = size.width / CGFloat(barCount)

            for index in 0..<barCount {
                let magnitude: CGFloat
                if magnitudes.isEmpty || !isPlaying {
                    magnitude = 0.04
                } else {
                    let bucket = Int(Double(index) / Double(barCount) * Double(magnitudes.count))
                    let clamped = min(max(0, bucket), magnitudes.count - 1)
                    magnitude = CGFloat(min(1.0, magnitudes[clamped]))
                }

                let height = max(4, magnitude * size.height)
                let x = CGFloat(index) * barWidth
                let rect = CGRect(x: x + 1, y: size.height - height, width: barWidth - 2, height: height)
                let opacity = isPlaying ? 0.75 : 0.3
                context.fill(Path(roundedRect: rect, cornerRadius: 2), with: .color(.mint.opacity(opacity)))
            }
        }
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10))
    }
}
