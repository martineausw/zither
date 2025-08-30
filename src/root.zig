pub const Tensor = @import("tensor.zig").Tensor;
pub const TensorView = @import("tensor_view.zig").TensorView;

test {
    @import("std").testing.refAllDecls(@This());
}
