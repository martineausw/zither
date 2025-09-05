const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

const duct = @import("duct");

const Tensor = @import("../../../tensor.zig").Tensor;
const root_utils = @import("../../../utils.zig");

pub fn set(comptime T: type) type {
    return struct {
        pub fn map(
            tensor: *Tensor(T),
            scalar: T,
            func: *const fn (
                scalar: T,
                elements: T,
                index: usize,
                data: struct { []const T, []const T },
            ) T,
        ) void {
            try duct.all.ops.scl.set(T).map(
                &tensor.*.buffer,
                scalar,
                func,
            );
        }

        pub fn add(
            tensor: *Tensor(T),
            scalar: T,
        ) void {
            return map(
                tensor,
                scalar,
                duct.all.ops.scl_func(T, Tensor(T)),
            );
        }

        pub fn sub(
            tensor: *Tensor(T),
            scalar: T,
        ) void {
            return map(
                tensor,
                scalar,
                duct.all.ops.scl_func(T, Tensor(T)).sub,
            );
        }

        pub fn mul(
            tensor: *Tensor(T),
            scalar: T,
        ) void {
            return map(
                tensor,
                scalar,
                duct.all.ops.scl_func(T, Tensor(T)).mul,
            );
        }

        pub fn div(
            tensor: *Tensor(T),
            scalar: T,
        ) void {
            return map(
                tensor,
                scalar,
                duct.all.ops.scl_func(T, Tensor(T)).div,
            );
        }

        pub fn divFloor(
            tensor: *Tensor(T),
            scalar: T,
        ) void {
            return map(
                tensor,
                scalar,
                duct.all.ops.scl_func(T, Tensor(T)).divFloor,
            );
        }

        pub fn divCeil(
            tensor: *Tensor(T),
            scalar: T,
        ) void {
            return map(
                tensor,
                scalar,
                duct.all.ops.scl_func(T, Tensor(T)).divCeil,
            );
        }
    };
}
