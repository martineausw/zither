const duct = @import("duct");
pub const elm = @import("all/elm.zig");
pub const scl = @import("all/scl.zig");

const Tensor = @import("../tensor.zig").Tensor;
const root_utils = @import("../utils.zig");

pub fn forEach(
    comptime T: type,
    tensor: Tensor(T),
    func: *const fn (element: T, index: usize, tensor: *const Tensor(T)) void,
) void {
    duct.all.forEach(tensor.*.buffer, func);
}

pub fn some(
    comptime T: type,
    tensor: Tensor(T),
    func: *const fn (element: T, index: usize, tensor: *const Tensor(T)) bool,
) bool {
    duct.all.some(tensor.*.buffer, func);
}

pub fn every(
    comptime T: type,
    tensor: Tensor(T),
    func: *const fn (element: T, index: usize, tensor: *const Tensor(T)) bool,
) bool {
    duct.all.every(tensor.*.buffer, func);
}

test {
    const testing = @import("std").testing;
    // try testing.expect(false);
    testing.refAllDecls(@This());

    const indices_0 = try root_utils.dimIndex(testing.allocator, &.{ 9, 3, 1 }, 0);
    defer testing.allocator.free(indices_0);
    const indices_1 = try root_utils.dimIndex(testing.allocator, &.{ 9, 3, 1 }, 26);
    defer testing.allocator.free(indices_1);
    try testing.expectEqualSlices(usize, &.{ 0, 0, 0 }, indices_0);
    try testing.expectEqualSlices(usize, &.{ 2, 2, 2 }, indices_1);
}
