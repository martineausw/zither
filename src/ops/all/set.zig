const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

const duct = @import("duct");
const Tensor = @import("../../tensor.zig").Tensor;
const utils = @import("../../utils.zig");

pub fn set(comptime T: type) type {
    return struct {
        pub fn map(
            dest: *Tensor(T),
            aux: *const Tensor(T),
            func: *const fn (
                elements: struct { T, T },
                index: usize,
                data: struct { []const T, []const T },
            ) T,
        ) void {
            if (dest.*.shape != aux.*.shape) return error.MismatchedShape;

            duct.all.ops.elm.set.map(
                &dest.*.buffer,
                aux.*.buffer,
                func,
            );
        }

        pub fn add(
            dest: *Tensor(T),
            aux: *const Tensor(T),
        ) void {
            map(
                dest,
                aux,
                duct.all.ops.elm_func(T, Tensor(T), Tensor(T)).add,
            );
        }

        pub fn sub(
            dest: *Tensor(T),
            aux: *const Tensor(T),
        ) void {
            map(
                dest,
                aux,
                duct.all.ops.elm_func(T, Tensor(T), Tensor(T)).sub,
            );
        }

        pub fn mul(
            dest: *const Tensor(T),
            aux: *const Tensor(T),
        ) void {
            map(
                T,
                dest,
                aux,
                duct.all.ops.elm_func(T, Tensor(T), Tensor(T)).mul,
            );
        }

        pub fn div(
            dest: *const Tensor(T),
            aux: *const Tensor(T),
        ) void {
            map(
                T,
                dest,
                aux,
                duct.all.ops.elm.elm_func(T, Tensor(T), Tensor(T)).div,
            );
        }

        pub fn divFloor(
            dest: *const Tensor(T),
            aux: *const Tensor(T),
        ) void {
            map(
                T,
                dest,
                aux,
                duct.all.ops.elm.elm_func(T, Tensor(T), Tensor(T)).divFloor,
            );
        }

        pub fn divCeil(
            dest: *const Tensor(T),
            aux: *const Tensor(T),
        ) void {
            map(
                dest,
                aux,
                duct.all.ops.elm_func(T, Tensor(T), Tensor(T)).divCeil,
            );
        }
    };
}
