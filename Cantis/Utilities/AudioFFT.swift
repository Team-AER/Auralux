import Accelerate
import AVFoundation
import Foundation

struct AudioFFT {
    /// Computes magnitude spectrum from audio samples using a real FFT via Accelerate.
    /// Returns `fftSize/2` magnitude bins normalized to 0...~1 range.
    static func magnitudes(samples: [Float], fftSize: Int = 1024) -> [Float] {
        guard samples.count >= fftSize, fftSize.isPowerOfTwo else { return [] }

        let log2n = vDSP_Length(log2(Double(fftSize)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else { return [] }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        var windowedSamples = Array(samples.prefix(fftSize))

        // Apply Hann window to reduce spectral leakage
        var window = [Float](repeating: 0, count: fftSize)
        vDSP_hann_window(&window, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))
        vDSP_vmul(windowedSamples, 1, window, 1, &windowedSamples, 1, vDSP_Length(fftSize))

        let halfSize = fftSize / 2
        var realPart = [Float](repeating: 0, count: halfSize)
        var imagPart = [Float](repeating: 0, count: halfSize)
        var magnitudes = [Float](repeating: 0, count: halfSize)

        // Pack interleaved real data into split complex form
        realPart.withUnsafeMutableBufferPointer { realPtr in
            imagPart.withUnsafeMutableBufferPointer { imagPtr in
                guard let realBase = realPtr.baseAddress,
                      let imagBase = imagPtr.baseAddress else { return }

                var splitComplex = DSPSplitComplex(realp: realBase, imagp: imagBase)
                windowedSamples.withUnsafeBufferPointer { inputPtr in
                    guard let inputBase = inputPtr.baseAddress else { return }
                    inputBase.withMemoryRebound(to: DSPComplex.self, capacity: halfSize) { complexPtr in
                        vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(halfSize))
                    }
                }

                // Perform forward FFT
                vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))

                // Compute magnitudes
                vDSP_zvmags(&splitComplex, 1, &magnitudes, 1, vDSP_Length(halfSize))
            }
        }

        // Scale and take square root for amplitude
        var scaleFactor: Float = 1.0 / Float(fftSize)
        vDSP_vsmul(magnitudes, 1, &scaleFactor, &magnitudes, 1, vDSP_Length(halfSize))
        vvsqrtf(&magnitudes, magnitudes, [Int32(halfSize)])

        // Zero DC bin (bin 0) — it represents the mean offset of the signal
        // and inflates low-frequency bars in the spectrum visualizer.
        if !magnitudes.isEmpty { magnitudes[0] = 0 }

        // Normalize to 0...1 range based on peak
        var peak: Float = 0
        vDSP_maxv(magnitudes, 1, &peak, vDSP_Length(halfSize))
        if peak > 0 {
            var invPeak = 1.0 / peak
            vDSP_vsmul(magnitudes, 1, &invPeak, &magnitudes, 1, vDSP_Length(halfSize))
        }

        return magnitudes
    }
}

private extension Int {
    var isPowerOfTwo: Bool {
        self > 0 && (self & (self - 1)) == 0
    }
}
