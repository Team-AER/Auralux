import SwiftUI

struct WaveformView: View {
    var progress: Double
    var samples: [Float]
    var onSeek: ((Double) -> Void)?

    @State private var scrubFraction: Double? = nil

    private var displayProgress: Double { scrubFraction ?? progress }

    var body: some View {
        GeometryReader { geo in
            Canvas { context, size in
                let baseline = size.height / 2
                let barWidth: CGFloat = 3
                let spacing: CGFloat = 2
                let count = Int(size.width / (barWidth + spacing))
                let filled = Int(Double(count) * displayProgress)

                for index in 0..<max(1, count) {
                    let amplitude: Double
                    if samples.isEmpty {
                        amplitude = 0.15
                    } else {
                        let sampleIndex = Int(Double(index) / Double(count) * Double(samples.count))
                        let clamped = min(max(0, sampleIndex), samples.count - 1)
                        amplitude = Double(samples[clamped])
                    }

                    let barHeight = max(4, amplitude * (size.height * 0.9))
                    let x = CGFloat(index) * (barWidth + spacing)
                    let rect = CGRect(x: x, y: baseline - barHeight / 2, width: barWidth, height: barHeight)
                    let color: Color = index <= filled ? .accentColor : .secondary.opacity(0.3)
                    context.fill(Path(roundedRect: rect, cornerRadius: 2), with: .color(color))
                }

                if scrubFraction != nil {
                    let lineX = displayProgress * size.width
                    var path = Path()
                    path.move(to: CGPoint(x: lineX, y: 0))
                    path.addLine(to: CGPoint(x: lineX, y: size.height))
                    context.stroke(path, with: .color(.white.opacity(0.85)), lineWidth: 2)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 12))
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        scrubFraction = max(0, min(1, value.location.x / geo.size.width))
                    }
                    .onEnded { value in
                        let fraction = max(0, min(1, value.location.x / geo.size.width))
                        onSeek?(fraction)
                        scrubFraction = nil
                    }
            )
        }
    }
}
