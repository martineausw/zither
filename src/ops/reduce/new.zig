const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

const duct = @import("duct");

const Tensor = @import("../../tensor.zig").Tensor;
const base_utils = @import("../../utils.zig");
const ops_utils = @import("../utils.zig");

pub fn new(comptime T: type) type {
    return struct {
        pub fn reduce(
            allocator: Allocator,
            initial: T,
            tensor: Tensor(T),
            axes: []const usize,
            func: *const fn (
                accumulator: T,
                element: T,
                indices: []const usize,
                tensor: *const Tensor(T),
            ) T,
        ) !Tensor(T) {

            // Create valid shape for output tensor
            const shape = try ops_utils.reduce(T).createShape(
                allocator,
                tensor.shape,
                axes,
            );

            // Create output tensor
            var new_tensor: Tensor(T) = try .init(allocator, shape);

            allocator.free(shape);

            // Create accessors for operand tensors
            var new_indices = try duct.new.zeroes(allocator, usize, new_tensor.rank());
            var indices = try duct.new.zeroes(allocator, usize, tensor.rank());

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

                new_tensor.set(new_indices, ops_utils.reduce(T).calculateElement(
                    initial,
                    tensor,
                    axes,
                    &indices,
                    func,
                    0,
                ));
            }

            allocator.free(new_indices);
            allocator.free(indices);

            return new_tensor;
        }

        pub fn sum(
            allocator: Allocator,
            tensor: Tensor(T),
            axes: []const usize,
        ) !Tensor(T) {
            return reduce(
                allocator,
                0,
                tensor,
                axes,
                ops_utils.reduce(T).sum,
            );
        }

        pub fn product(
            allocator: Allocator,
            tensor: Tensor(T),
            axes: []const usize,
        ) !Tensor(T) {
            return reduce(
                allocator,
                1,
                tensor,
                axes,
                ops_utils.reduce(T).product,
            );
        }
    };
}

test "sum reduction" {
    var A: Tensor(f32) = try .arange(testing.allocator, 1, 10, 1);
    defer A.deinit();

    try A.reshape(&.{ 3, 3 });

    const A_row_sum = try new(f32).sum(testing.allocator, A, &.{1});
    defer A_row_sum.deinit();

    const A_col_sum = try new(f32).sum(testing.allocator, A, &.{0});
    defer A_col_sum.deinit();

    const A_sum = try new(f32).sum(testing.allocator, A, &.{ 0, 1 });
    defer A_sum.deinit();

    try testing.expectEqualSlices(usize, &.{3}, A_row_sum.shape);
    try testing.expectEqualSlices(f32, &.{ 6, 15, 24 }, A_row_sum.buffer);
    try testing.expectEqualSlices(usize, &.{3}, A_col_sum.shape);
    try testing.expectEqualSlices(f32, &.{ 12, 15, 18 }, A_col_sum.buffer);
    try testing.expectEqualSlices(usize, &.{}, A_sum.shape);
    try testing.expectEqualSlices(f32, &.{45}, A_sum.buffer);
}

test "product reduction" {
    testing.log_level = .debug;
    var A: Tensor(f32) = try .arange(testing.allocator, 1, 10, 1);
    defer A.deinit();

    try A.reshape(&.{ 3, 3 });

    const A_row_product = try new(f32).product(testing.allocator, A, &.{1});
    defer A_row_product.deinit();

    const A_col_product = try new(f32).product(testing.allocator, A, &.{0});
    defer A_col_product.deinit();

    const A_product = try new(f32).product(testing.allocator, A, &.{ 0, 1 });
    defer A_product.deinit();

    try testing.expectEqualSlices(usize, &.{3}, A_row_product.shape);
    try testing.expectEqualSlices(f32, &.{ 6, 120, 504 }, A_row_product.buffer);
    try testing.expectEqualSlices(usize, &.{3}, A_col_product.shape);
    try testing.expectEqualSlices(f32, &.{ 28, 80, 162 }, A_col_product.buffer);
    try testing.expectEqualSlices(usize, &.{}, A_product.shape);
    try testing.expectEqualSlices(f32, &.{362880}, A_product.buffer);
}
