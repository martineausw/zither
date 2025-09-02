const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const Tensor = @import("tensor.zig").Tensor;
const utils = @import("utils.zig");
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
    indices_1: *[]usize,
    axes_1: []const usize,
) void {
    var index: usize = 0;

    // Splice tensor accessor indices from output_indices
    for (0..indices_0.*.len) |index_0| {
        if (duct.get.indexOf(axes_0, index_0)) |_| continue;
        indices_0.*[index_0] = indices[index];
        index += 1;
    }

    // Splice tensor accessor indices from output_indices
    for (0..indices_1.*.len) |index_1| {
        if (duct.get.indexOf(axes_1, index_1)) |_| continue;
        indices_1.*[index_1] = indices[index];
        index += 1;
    }
}

fn createShape(
    allocator: Allocator,
    shape_0: []const usize,
    axes_0: []const usize,
    shape_1: []const usize,
    axes_1: []const usize,
) Allocator.Error![]const usize {
    const len = (shape_0.len - axes_0.len) + (shape_1.len - axes_1.len);

    const shape = try allocator.alloc(
        usize,
        if (len == 0) 1 else len,
    );

    var dim: usize = 0;
    var index_0: usize = 0;
    for (0..shape_0.len) |dim_0| {
        if (index_0 < axes_0.len and dim_0 == axes_0[index_0]) {
            index_0 += 1;
            continue;
        }
        shape[dim] = shape_0[dim_0];
        dim += 1;
    }

    var index_1: usize = 0;
    for (0..shape_1.len) |dim_1| {
        if (index_1 < axes_1.len and dim_1 == axes_1[index_1]) {
            index_1 += 1;
            continue;
        }

        if (dim < shape.len) {
            shape[dim] = shape_1[dim_1];
            dim += 1;
        }
    }

    return shape;
}

pub fn contraction(
    allocator: Allocator,
    comptime T: type,
    tensor_0: Tensor(T),
    axes_0: []const usize,
    tensor_1: Tensor(T),
    axes_1: []const usize,
) !Tensor(T) {

    // Validate shared axes
    for (0..axes_0.len) |index| {
        if (tensor_0.shape[axes_0[index]] != tensor_1.shape[axes_1[index]]) return error.MismatchedAxis;
    }

    // Create valid shape for output tensor
    const shape = try createShape(
        allocator,
        tensor_0.shape,
        axes_0,
        tensor_1.shape,
        axes_1,
    );

    // Create output tensor
    var tensor: Tensor(T) = try .init(allocator, shape);

    allocator.free(shape);

    // Create accessors for operand tensors
    var indices = try duct.new.zeroes(allocator, usize, tensor.rank());
    var indices_0 = try duct.new.zeroes(allocator, usize, tensor_0.rank());
    var indices_1 = try duct.new.zeroes(allocator, usize, tensor_1.rank());

    for (0..utils.flatLen(tensor.shape)) |it| {
        try incrementIndices(
            tensor.strides,
            &indices,
            it,
        );

        mapAccessorIndices(
            indices,
            &indices_0,
            axes_0,
            &indices_1,
            axes_1,
        );

        tensor.set(indices, calculateElement(
            T,
            tensor_0,
            axes_0,
            &indices_0,
            tensor_1,
            axes_1,
            &indices_1,
            0,
        ));
    }

    allocator.free(indices);
    allocator.free(indices_0);
    allocator.free(indices_1);

    return tensor;
}

fn calculateElement(
    comptime T: type,
    tensor_0: Tensor(T),
    axes_0: []const usize,
    indices_0: *[]usize,
    tensor_1: Tensor(T),
    axes_1: []const usize,
    indices_1: *[]usize,
    depth: usize,
) T {
    var result: T = 0;
    if (depth == axes_0.len - 1) {
        for (0..tensor_0.shape[axes_0[depth]]) |dim| {
            indices_0.*[axes_0[depth]] = dim;
            indices_1.*[axes_1[depth]] = dim;
            result += tensor_0.at(indices_0.*) * tensor_1.at(indices_1.*);
        }
    } else {
        for (0..tensor_0.shape[axes_0[depth]]) |dim| {
            indices_0.*[axes_0[depth]] = dim;
            indices_1.*[axes_1[depth]] = dim;
            result += calculateElement(
                T,
                tensor_0,
                axes_0,
                indices_0,
                tensor_1,
                axes_1,
                indices_1,
                depth + 1,
            );
        }
    }

    return result;
}

test "contraction shape" {
    const A: Tensor(f32) = try .init(testing.allocator, &.{ 2, 4, 3 });
    defer A.deinit();
    const B: Tensor(f32) = try .init(testing.allocator, &.{ 4, 3, 3 });
    defer B.deinit();

    const actual = try createShape(testing.allocator, A.shape, &.{1}, B.shape, &.{0});
    defer testing.allocator.free(actual);

    try testing.expectEqualSlices(usize, &.{ 2, 3, 3, 3 }, actual);
}

test "identity" {
    const A: Tensor(f32) = try .identity(testing.allocator, 3);
    defer A.deinit();
    const B: Tensor(f32) = try .arange(testing.allocator, 0, 3, 1);
    defer B.deinit();

    const AA = try contraction(
        testing.allocator,
        f32,
        A,
        &.{1},
        A,
        &.{1},
    );
    defer AA.deinit();

    const AB = try contraction(
        testing.allocator,
        f32,
        A,
        &.{1},
        B,
        &.{0},
    );
    defer AB.deinit();

    try testing.expectEqualSlices(f32, A.buffer, AA.buffer);
    try testing.expectEqualSlices(f32, B.buffer, AB.buffer);
}

test "matrix" {
    var A: Tensor(f32) = try .arange(testing.allocator, 1, 10, 1);
    defer A.deinit();

    try testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, A.buffer);

    try A.reshape(&.{ 3, 3 });

    try testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, A.buffer);

    try testing.expectEqualSlices(usize, &.{ 3, 3 }, A.shape);

    const AA = try contraction(
        testing.allocator,
        f32,
        A,
        &.{0},
        A,
        &.{0},
    );
    defer AA.deinit();

    try testing.expectEqualSlices(f32, &.{ 66, 78, 90, 78, 93, 108, 90, 108, 126 }, AA.buffer);
}

test "cube" {
    const A: Tensor(f32) = try .ones(testing.allocator, &.{ 3, 3, 3 });
    defer A.deinit();

    try testing.expectEqual(27, A.buffer.len);

    for (0..A.buffer.len) |index| {
        try testing.expectEqual(1, A.buffer[index]);
    }

    const AA = try contraction(
        testing.allocator,
        f32,
        A,
        &.{ 0, 1 },
        A,
        &.{ 0, 1 },
    );
    defer AA.deinit();

    try testing.expectEqualSlices(usize, &.{ 3, 3, 3 }, A.shape);
    try testing.expectEqual(9, AA.buffer.len);
    try testing.expectEqualSlices(usize, &.{ 3, 3 }, AA.shape);
    try testing.expectEqualSlices(f32, &.{ 9, 9, 9, 9, 9, 9, 9, 9, 9 }, AA.buffer);
}

test "cube arithmetic" {
    var A: Tensor(f128) = try .arange(testing.allocator, 1, 28, 1);
    defer A.deinit();

    // Buffer sanity check
    for (0..A.buffer.len) |index| {
        try testing.expectEqual(@as(f32, @floatFromInt(index + 1)), A.buffer[index]);
    }

    try A.reshape(&.{ 3, 3, 3 });

    // Reshape sanity check
    for (0..A.buffer.len) |index| {
        try testing.expectEqual(@as(f32, @floatFromInt(index + 1)), A.buffer[index]);
    }

    // Index sanity check
    var value: f128 = 1;
    for (0..3) |z| {
        for (0..3) |y| {
            for (0..3) |x| {
                try testing.expectEqual(value, A.at(&.{ z, y, x }));
                value += 1;
            }
        }
    }

    // 2061  2178  2295
    // 2178  2304  2430
    // 2295  2430  2565
    var AA_expect: Tensor(f128) = try .init(testing.allocator, &.{ 3, 3 });
    defer AA_expect.deinit();
    AA_expect.set(&.{ 0, 0 }, 2061);
    AA_expect.set(&.{ 0, 1 }, 2178);
    AA_expect.set(&.{ 0, 2 }, 2295);
    AA_expect.set(&.{ 1, 0 }, 2178);
    AA_expect.set(&.{ 1, 1 }, 2304);
    AA_expect.set(&.{ 1, 2 }, 2430);
    AA_expect.set(&.{ 2, 0 }, 2295);
    AA_expect.set(&.{ 2, 1 }, 2430);
    AA_expect.set(&.{ 2, 2 }, 2565);

    const AA = try contraction(
        testing.allocator,
        f128,
        A,
        &.{ 0, 1 },
        A,
        &.{ 0, 1 },
    );
    defer AA.deinit();

    try testing.expectEqualSlices(f128, AA_expect.buffer, AA.buffer);
}
