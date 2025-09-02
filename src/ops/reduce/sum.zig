const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const Tensor = @import("../../tensor.zig").Tensor;
const utils = @import("../../utils.zig");
const duct = @import("duct");

fn incrementIndices(strides: []const usize, indices: *[]usize, iteration: usize) !void {
    var value = iteration;

    for (0..indices.len) |index| {
        indices.*[index] = @intCast(@divFloor(value, strides[index]));
        value -= strides[index] * @divFloor(value, strides[index]);
    }
}

fn mapAccessorIndices(
    indices: []const usize,
    indices_0: *[]usize,
    axes_0: []const usize,
) void {
    var index: usize = 0;

    // Splice tensor accessor indices from output_indices
    for (0..indices_0.*.len) |index_0| {
        if (duct.get.indexOf(axes_0, index_0)) |_| continue;
        indices_0.*[index_0] = indices[index];
        index += 1;
    }
}

fn createShape(
    allocator: Allocator,
    shape: []const usize,
    axes: []const usize,
) Allocator.Error![]const usize {
    const len = (shape.len - axes.len);

    const new_shape = try allocator.alloc(
        usize,
        len,
    );

    var dim: usize = 0;
    var index_0: usize = 0;
    for (0..shape.len) |dim_0| {
        if (index_0 < axes.len and dim_0 == axes[index_0]) {
            index_0 += 1;
            continue;
        }
        new_shape[dim] = shape[dim_0];
        dim += 1;
    }

    return new_shape;
}

pub fn sum(
    allocator: Allocator,
    comptime T: type,
    tensor: Tensor(T),
    axes: []const usize,
) !Tensor(T) {

    // Create valid shape for output tensor
    const shape = try createShape(
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

    for (0..utils.flatLen(new_tensor.shape)) |it| {
        try incrementIndices(
            new_tensor.strides,
            &new_indices,
            it,
        );

        mapAccessorIndices(
            new_indices,
            &indices,
            axes,
        );

        new_tensor.set(new_indices, calculateElement(
            T,
            tensor,
            axes,
            &indices,
            0,
        ));
    }

    allocator.free(new_indices);
    allocator.free(indices);

    return new_tensor;
}

fn calculateElement(
    comptime T: type,
    tensor: Tensor(T),
    axes: []const usize,
    indices: *[]usize,
    depth: usize,
) T {
    var result: T = 0;
    if (depth == axes.len - 1) {
        for (0..tensor.shape[axes[depth]]) |dim| {
            indices.*[axes[depth]] = dim;
            result += tensor.at(indices.*);
        }
    } else {
        for (0..tensor.shape[axes[depth]]) |dim| {
            indices.*[axes[depth]] = dim;
            result += calculateElement(
                T,
                tensor,
                axes,
                indices,
                depth + 1,
            );
        }
    }

    return result;
}

test "sum reduction" {
    var A: Tensor(f32) = try .arange(testing.allocator, 1, 10, 1);
    defer A.deinit();

    try A.reshape(&.{ 3, 3 });

    const A_row_sum = try sum(testing.allocator, f32, A, &.{1});
    defer A_row_sum.deinit();

    const A_col_sum = try sum(testing.allocator, f32, A, &.{0});
    defer A_col_sum.deinit();

    const A_sum = try sum(testing.allocator, f32, A, &.{ 0, 1 });
    defer A_sum.deinit();

    try testing.expectEqualSlices(usize, &.{3}, A_row_sum.shape);
    try testing.expectEqualSlices(f32, &.{ 6, 15, 24 }, A_row_sum.buffer);
    try testing.expectEqualSlices(usize, &.{3}, A_col_sum.shape);
    try testing.expectEqualSlices(f32, &.{ 12, 15, 18 }, A_col_sum.buffer);
    try testing.expectEqualSlices(usize, &.{}, A_sum.shape);
    try testing.expectEqualSlices(f32, &.{45}, A_sum.buffer);
}
