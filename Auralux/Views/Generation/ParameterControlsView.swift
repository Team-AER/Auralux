import SwiftUI

struct ParameterControlsView: View {
    @Binding var duration: Double
    @Binding var variance: Double
    @Binding var seedText: String

    var body: some View {
        GroupBox("Parameters") {
            VStack(alignment: .leading, spacing: 12) {
                SliderControl(label: "Duration", value: $duration, range: 10...180, unit: "sec")
                SliderControl(label: "Variance", value: $variance, range: 0...1, unit: "")

                HStack {
                    Text("Seed")
                        .frame(width: 100, alignment: .leading)
                    TextField("Random", text: $seedText)
                        .textFieldStyle(.roundedBorder)
                        .frame(maxWidth: 240)
                        .onChange(of: seedText) { _, newValue in
                            let digits = newValue.filter(\.isNumber)
                            if digits != newValue { seedText = digits }
                        }
                }
            }
        }
    }
}
