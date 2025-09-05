const duct = @import("duct");

const Tensor = @import("../../tensor.zig").Tensor;
const root_utils = @import("../../utils.zig");

pub fn set(comptime T: type) type {
    return struct {
        pub fn map(
            tensor: *Tensor(T),
            func: *const fn (element: T, index: usize, tensor: []const T) T,
        ) void {
            duct.all.set.map(&tensor.*.buffer, func);
        }
    };
}
