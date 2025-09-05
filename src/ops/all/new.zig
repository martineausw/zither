const Allocator = @import("std").mem.Allocator;
const duct = @import("duct");

const Tensor = @import("../../tensor.zig");
const root_utils = @import("../../utils.zig");

pub fn new(comptime T: type) type {
    return struct {
        pub fn map(
            allocator: Allocator,
            tensor: Tensor(T),
            func: *const fn (element: T, index: usize, tensor: *const Tensor(T)) T,
        ) !Tensor(T) {
            return .{
                .buffer = try duct.all.new.map(allocator, tensor.buffer, func),
                .shape = duct.new.copy(allocator, tensor.shape),
                .strides = duct.new.copy(allocator, tensor.strides),
                .allocator = allocator,
            };
        }
    };
}
