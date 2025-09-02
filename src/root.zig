pub const Tensor = @import("tensor.zig").Tensor;
// pub const TensorView = @import("tensor_view.zig").TensorView;
pub const contraction = @import("contraction.zig").contraction;

test {
    @import("std").testing.refAllDecls(@This());
}
