@preconcurrency import Foundation
@preconcurrency import MLX
@preconcurrency import MLXNN

class MyMod: Module {
    var table: MLXArray
    override init() {
        self.table = MLXArray.zeros([10])
        super.init()
    }
}

func testParamKeys() {
    let m = MyMod()
    let keys = m.parameters().keys
    print("Keys in MyMod: \(keys)")
}
