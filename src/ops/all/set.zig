const duct = @import("duct");

const Tensor = @import("../../tensor.zig");
const root_utils = @import("../../utils.zig");

pub fn set(comptime T: type) type {
    return struct {
        pub fn map(
            tensor: *Tensor(T),
            func: *const fn (element: T, index: usize, tensor: *const Tensor(T)) T,
        ) void {
            duct.all.set.map(&tensor.*.buffer, func);
        }
    };
}
