const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

const duct = @import("duct");

const Tensor = @import("../../tensor.zig").Tensor;
const base_utils = @import("../../utils.zig");
const ops_utils = @import("../utils.zig");

pub fn set(comptime T: type) type {
    return struct {
        pub fn reduce(
            initial: T,
            tensor: *Tensor(T),
            axes: []const usize,
            func: *const fn (
                accumulator: T,
                element: T,
                indices: []const usize,
                tensor: *const Tensor(T),
            ) T,
        ) !void {

            // Create valid shape for output tensor
            const shape = try ops_utils.reduce(T).createShape(
                tensor.*.allocator,
                tensor.*.shape,
                axes,
            );

            // Create output tensor
            var new_tensor: Tensor(T) = try .init(tensor.*.allocator, shape);

            tensor.*.allocator.free(shape);

            // Create accessors for operand tensors
            var new_indices = try duct.new.zeroes(tensor.*.allocator, usize, new_tensor.rank());
            var indices = try duct.new.zeroes(tensor.*.allocator, usize, tensor.rank());

            for (0..base_utils.flatLen(new_tensor.shape)) |it| {
                try ops_utils.incrementIndices(
                    new_tensor.strides,
                    &new_indices,
                    it,
                );

                ops_utils.reduce(T).mapAccessorIndices(
                    new_indices,
                    &indices,
                    axes,
                );

                new_tensor.set(new_indices, ops_utils.calculateElement(
                    initial,
                    tensor,
                    axes,
                    &indices,
                    func,
                    0,
                ));
            }

            tensor.*.allocator.free(new_indices);
            tensor.*.allocator.free(indices);

            tensor.*.deinit();

            tensor.*.buffer = new_tensor.buffer;
            tensor.*.shape = new_tensor.shape;
            tensor.*.strides = new_tensor.strides;
        }

        pub fn sum(
            tensor: *Tensor(T),
            axes: []const usize,
        ) !void {
            reduce(
                0,
                tensor,
                axes,
                ops_utils.reduce_func(T).sum,
            );
        }

        pub fn product(
            tensor: *Tensor(T),
            axes: []const usize,
        ) !void {
            return reduce(
                1,
                tensor,
                axes,
                ops_utils.reduce_func(T).product,
            );
        }
    };
}

test "sum reduction" {
    var A: Tensor(f32) = try .arange(testing.allocator, 1, 10, 1);
    defer A.deinit();
    var B: Tensor(f32) = try .arange(testing.allocator, 1, 10, 1);
    defer B.deinit();
    var C: Tensor(f32) = try .arange(testing.allocator, 1, 10, 1);
    defer C.deinit();

    try A.reshape(&.{ 3, 3 });
    try B.reshape(&.{ 3, 3 });
    try C.reshape(&.{ 3, 3 });

    try set(f32).sum(testing.allocator, &A, &.{1});
    try set(f32).sum(testing.allocator, &B, &.{0});
    try set(f32).sum(testing.allocator, &C, &.{ 0, 1 });

    try testing.expectEqualSlices(usize, &.{3}, A.shape);
    try testing.expectEqualSlices(f32, &.{ 6, 15, 24 }, A.buffer);
    try testing.expectEqualSlices(usize, &.{3}, B.shape);
    try testing.expectEqualSlices(f32, &.{ 12, 15, 18 }, B.buffer);
    try testing.expectEqualSlices(usize, &.{}, C.shape);
    try testing.expectEqualSlices(f32, &.{45}, C.buffer);
}

test "product reduction" {
    var A: Tensor(f32) = try .arange(testing.allocator, 1, 10, 1);
    defer A.deinit();
    var B: Tensor(f32) = try .arange(testing.allocator, 1, 10, 1);
    defer B.deinit();
    var C: Tensor(f32) = try .arange(testing.allocator, 1, 10, 1);
    defer C.deinit();

    try A.reshape(&.{ 3, 3 });
    try B.reshape(&.{ 3, 3 });
    try C.reshape(&.{ 3, 3 });

    try set(f32).product(&A, &.{1});
    try set(f32).product(&B, &.{0});
    try set(f32).product(&C, &.{ 0, 1 });

    try testing.expectEqualSlices(usize, &.{3}, A.shape);
    try testing.expectEqualSlices(f32, &.{ 6, 120, 504 }, A.buffer);
    try testing.expectEqualSlices(usize, &.{3}, B.shape);
    try testing.expectEqualSlices(f32, &.{ 28, 80, 162 }, B.buffer);
    try testing.expectEqualSlices(usize, &.{}, C.shape);
    try testing.expectEqualSlices(f32, &.{362880}, C.buffer);
}
