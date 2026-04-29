import MLX
import Foundation

/// Direct port of `pack_sequences` from `modeling_acestep_v15_turbo.py:141-172`.
///
/// Given two `[B, L_i, D]` hidden-state tensors with corresponding `[B, L_i]` masks,
/// produces a single packed `[B, L1+L2, D]` sequence in which all valid (mask=1) tokens
/// come first within each batch row, followed by padding (mask=0). Returns the new mask
/// `[B, L1+L2]` indicating valid positions in the packed output.
enum PackSequences {

    static func pack(
        _ h1: MLXArray, _ h2: MLXArray,
        _ m1: MLXArray, _ m2: MLXArray
    ) -> (hidden: MLXArray, mask: MLXArray) {
        let hCat = concatenated([h1, h2], axis: 1)         // [B, L1+L2, D]
        let mCat = concatenated([m1, m2], axis: 1)         // [B, L1+L2]

        let B = hCat.shape[0]
        let L = hCat.shape[1]
        let D = hCat.shape[2]

        // argSort is ascending; negate (cast to int first) to get descending → 1s before 0s.
        let mInt    = mCat.asType(.int32)
        let sortKey = -mInt
        let sortIdx = argSort(sortKey, axis: 1)            // [B, L]

        // Gather hidden along axis 1 using sortIdx, broadcast across the D dim.
        let idxExpanded = broadcast(
            sortIdx.expandedDimensions(axis: -1),
            to: [B, L, D]
        )
        let hSorted = takeAlong(hCat, idxExpanded, axis: 1)

        // New mask = arange(L) < lengths(b)
        let lengths = mCat.sum(axis: 1).asType(.int32)     // [B]
        let arangeRow = MLXArray(Array(0..<L).map { Int32($0) }).reshaped([1, L])
        let arangeBL  = broadcast(arangeRow, to: [B, L])
        let lenBL     = broadcast(lengths.reshaped([B, 1]), to: [B, L])
        let newMask   = (arangeBL .< lenBL).asType(.int32)

        return (hSorted, newMask)
    }
}
