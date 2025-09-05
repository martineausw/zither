const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

const duct = @import("duct");

const Tensor = @import("../../tensor.zig").Tensor;
const root_utils = @import("../../utils.zig");
const ops_utils = @import("../utils.zig");

pub fn set(comptime T: type) type {
    return struct {
        pub fn contract(
            dest: *Tensor(T),
            dest_axes: []const usize,
            aux: Tensor(T),
            aux_axes: []const usize,
            func: fn (
                accumulator: T,
                elements: struct { T, T },
                indices: struct { []const usize, []const usize },
                tensors: struct { *const Tensor(T), *const Tensor(T) },
            ) T,
            initial_value: T,
        ) !void {

            // Validate shared axes
            for (0..dest_axes.len) |index| {
                if (dest.shape[dest_axes[index]] != aux.shape[aux_axes[index]]) return error.MismatchedAxis;
            }

            // Create valid shape for output tensor
            const shape = try ops_utils.contract(T).createShape(
                dest.*.allocator,
                dest.*.shape,
                dest_axes,
                aux.shape,
                aux_axes,
            );

            // Create output tensor
            var tensor: Tensor(T) = try .init(dest.*.allocator, shape);

            dest.*.allocator.free(shape);

            // Create accessors for operand tensors
            var indices = try duct.new.zeroes(dest.*.allocator, usize, tensor.rank());
            var indices_0 = try duct.new.zeroes(dest.*.allocator, usize, dest.rank());
            var indices_1 = try duct.new.zeroes(dest.*.allocator, usize, aux.rank());

            for (0..root_utils.flatLen(tensor.shape)) |it| {
                try ops_utils.incrementIndices(
                    tensor.strides,
                    &indices,
                    it,
                );

                ops_utils.contract(T).mapAccessorIndices(
                    indices,
                    &indices_0,
                    dest_axes,
                    &indices_1,
                    aux_axes,
                );

                tensor.set(indices, ops_utils.contract(T).calculateElement(
                    func,
                    initial_value,
                    dest.*,
                    dest_axes,
                    &indices_0,
                    aux,
                    aux_axes,
                    &indices_1,
                    0,
                ));
            }

            dest.*.allocator.free(indices);
            dest.*.allocator.free(indices_0);
            dest.*.allocator.free(indices_1);

            dest.*.deinit();

            dest.*.buffer = tensor.buffer;
            dest.*.strides = tensor.strides;
            dest.*.shape = tensor.shape;
        }

        pub fn tensordot(
            dest: *Tensor(T),
            dest_axes: []const usize,
            aux: Tensor(T),
            aux_axes: []const usize,
        ) !void {
            try contract(
                dest,
                dest_axes,
                aux,
                aux_axes,
                ops_utils.contract(T).tensordot,
                0,
            );
        }
    };
}

test "identity" {
    var A_0: Tensor(f32) = try .identity(testing.allocator, 3);
    defer A_0.deinit();

    var A_1: Tensor(f32) = try .identity(testing.allocator, 3);
    defer A_1.deinit();

    var B: Tensor(f32) = try .arange(testing.allocator, 0, 3, 1);
    defer B.deinit();

    try set(f32).tensordot(
        &A_0,
        &.{1},
        A_0,
        &.{1},
    );

    try testing.expectEqualSlices(f32, &.{ 1, 0, 0, 0, 1, 0, 0, 0, 1 }, A_0.buffer);

    try set(f32).tensordot(
        &A_1,
        &.{1},
        B,
        &.{0},
    );

    try testing.expectEqualSlices(f32, &.{ 0, 1, 2 }, A_1.buffer);
}

test "matrix" {
    var A: Tensor(f32) = try .arange(testing.allocator, 1, 10, 1);
    defer A.deinit();

    try A.reshape(&.{ 3, 3 });

    try set(f32).tensordot(
        &A,
        &.{0},
        A,
        &.{0},
    );

    try testing.expectEqualSlices(f32, &.{ 66, 78, 90, 78, 93, 108, 90, 108, 126 }, A.buffer);
}

test "cube" {
    var A: Tensor(f32) = try .ones(testing.allocator, &.{ 3, 3, 3 });
    defer A.deinit();

    try set(f32).tensordot(
        &A,
        &.{ 0, 1 },
        A,
        &.{ 0, 1 },
    );

    try testing.expectEqual(9, A.buffer.len);
    try testing.expectEqualSlices(usize, &.{ 3, 3 }, A.shape);
    try testing.expectEqualSlices(f32, &.{ 9, 9, 9, 9, 9, 9, 9, 9, 9 }, A.buffer);
}

test "cube arithmetic" {
    var A: Tensor(f128) = try .arange(testing.allocator, 1, 28, 1);
    defer A.deinit();

    try A.reshape(&.{ 3, 3, 3 });

    try set(f128).tensordot(
        &A,
        &.{ 0, 1 },
        A,
        &.{ 0, 1 },
    );

    try testing.expectEqualSlices(f128, &.{ 2061, 2178, 2295, 2178, 2304, 2430, 2295, 2430, 2565 }, A.buffer);
}
