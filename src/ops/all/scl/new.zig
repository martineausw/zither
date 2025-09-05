const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

const duct = @import("duct");

const Tensor = @import("../../../tensor.zig").Tensor;
const root_utils = @import("../../../utils.zig");

pub fn new(comptime T: type) type {
    return struct {
        pub fn map(
            allocator: Allocator,
            tensor: *const Tensor(T),
            scalar: T,
            func: *const fn (
                scalar: T,
                elements: T,
                index: usize,
                data: struct { []const T, []const T },
            ) T,
        ) !Tensor(T) {
            const buffer = try duct.all.ops.scl.new(T).map(
                allocator,
                tensor,
                scalar,
                func,
            );

            return .{
                .buffer = buffer,
                .shape = try duct.new.copy(allocator, tensor.shape),
                .strides = try root_utils.initStrides(allocator, tensor.shape),
                .allocator = allocator,
            };
        }

        pub fn add(
            allocator: Allocator,
            tensor: *const Tensor(T),
            scalar: T,
        ) !Tensor(T) {
            return map(
                allocator,
                tensor,
                scalar,
                duct.all.ops.scl_func(T, Tensor(T)),
            );
        }

        pub fn sub(
            allocator: Allocator,
            tensor: *const Tensor(T),
            scalar: T,
        ) !Tensor(T) {
            return map(
                allocator,
                tensor,
                scalar,
                duct.all.ops.scl_func(T, Tensor(T)).sub,
            );
        }

        pub fn mul(
            allocator: Allocator,
            tensor: *const Tensor(T),
            scalar: T,
        ) !Tensor(T) {
            return map(
                allocator,
                tensor,
                scalar,
                duct.all.ops.scl_func(T, Tensor(T)).mul,
            );
        }

        pub fn div(
            allocator: Allocator,
            tensor: *const Tensor(T),
            scalar: T,
        ) !Tensor(T) {
            return map(
                allocator,
                tensor,
                scalar,
                duct.all.ops.scl_func(T, Tensor(T)).div,
            );
        }

        pub fn divFloor(
            allocator: Allocator,
            tensor: *const Tensor(T),
            scalar: T,
        ) !Tensor(T) {
            return map(
                allocator,
                tensor,
                scalar,
                duct.all.ops.scl_func(T, Tensor(T)).divFloor,
            );
        }

        pub fn divCeil(
            allocator: Allocator,
            tensor: *const Tensor(T),
            scalar: T,
        ) !Tensor(T) {
            return map(
                allocator,
                tensor,
                scalar,
                duct.all.ops.scl_func(T, Tensor(T)).divCeil,
            );
        }
    };
}
