import MLX
import MLXNN
class MyMod: Module {
    @Parameter var table: MLXArray
    init() {
        self.table = MLXArray.zeros([10])
        super.init()
    }
}
let m = MyMod()
print(m.parameters().keys)
