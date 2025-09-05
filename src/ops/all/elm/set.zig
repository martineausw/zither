const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

const duct = @import("duct");

const Tensor = @import("../../../tensor.zig").Tensor;
const root_utils = @import("../../../utils.zig");

pub fn set(comptime T: type) type {
    return struct {
        pub fn map(
            dest: *Tensor(T),
            aux: Tensor(T),
            func: *const fn (
                elements: struct { T, T },
                index: usize,
                data: struct { []const T, []const T },
            ) T,
        ) void {
            if (dest.*.shape != aux.shape) return error.MismatchedShape;

            duct.all.ops.elm.set(T).map(
                dest.*.buffer,
                aux.buffer,
                func,
            );
        }

        pub fn add(
            dest: *Tensor(T),
            aux: Tensor(T),
        ) void {
            map(
                dest,
                aux,
                duct.all.ops.elm_func(T, []const T, []const T).add,
            );
        }

        pub fn sub(
            dest: *Tensor(T),
            aux: Tensor(T),
        ) void {
            map(
                dest,
                aux,
                duct.all.ops.elm_func(T, []T, []T).sub,
            );
        }

        pub fn mul(
            dest: *Tensor(T),
            aux: Tensor(T),
        ) void {
            map(
                T,
                dest.*,
                aux,
                duct.all.ops.elm_func(T, []T, []T).mul,
            );
        }

        pub fn div(
            dest: *Tensor(T),
            aux: Tensor(T),
        ) void {
            map(
                T,
                dest.*,
                aux,
                duct.all.ops.elm_func(T, []T, []T).div,
            );
        }

        pub fn divFloor(
            dest: *Tensor(T),
            aux: Tensor(T),
        ) void {
            map(
                T,
                dest.*,
                aux,
                duct.all.ops.elm.elm_func(T, []T, []T).divFloor,
            );
        }

        pub fn divCeil(
            dest: *Tensor(T),
            aux: Tensor(T),
        ) void {
            map(
                dest.*,
                aux,
                duct.all.ops.elm_func(T, []T, []T).divCeil,
            );
        }
    };
}
