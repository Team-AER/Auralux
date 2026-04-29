import SwiftUI

struct SliderControl: View {
    let label: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let unit: String
    var warningThreshold: Double? = nil
    var warningMessage: String? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                    .frame(width: 100, alignment: .leading)
                Slider(value: $value, in: range)
                    .tint(trackColor)
                Text(verbatim: valueLabel)
                    .font(.caption.monospacedDigit())
                    .frame(width: 70, alignment: .trailing)
                    .foregroundStyle(valueLabelStyle)
            }
            if isWarning, let warningMessage {
                Text(warningMessage)
                    .font(.caption2)
                    .foregroundStyle(.red)
                    .padding(.leading, 100)
                    .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.15), value: isWarning)
        .animation(.easeInOut(duration: 0.1), value: value)
    }

    /// True only when this slider is wired to a memory-impacting threshold and
    /// the user has crossed it. Sliders without a `warningThreshold` never warn.
    private var isWarning: Bool {
        guard let warningThreshold else { return false }
        return value > warningThreshold
    }

    /// Interpolates from blue toward red **only for sliders that affect memory**
    /// (i.e. those given a `warningThreshold`). All other sliders inherit the
    /// system accent so they don't visually imply danger.
    private var trackColor: Color? {
        guard let warningThreshold else { return nil }
        let span = max(range.upperBound - warningThreshold, 0.0001)
        let t = min(max((value - warningThreshold) / span, 0), 1)
        // Blue (0.0, 0.48, 1.0) → Red (1.0, 0.23, 0.19) via a perceptual lerp.
        let r = 0.0 + (1.0 - 0.0) * t
        let g = 0.48 + (0.23 - 0.48) * t
        let b = 1.0 + (0.19 - 1.0) * t
        return Color(red: r, green: g, blue: b)
    }

    private var valueLabelStyle: AnyShapeStyle {
        if isWarning, let trackColor {
            return AnyShapeStyle(trackColor)
        }
        return AnyShapeStyle(.secondary)
    }

    private var valueLabel: String {
        if unit == "" {
            return String(format: "%.2f", value)
        }
        return "\(Int(value)) \(unit)"
    }
}
