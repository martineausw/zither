const std = @import("std");
const Allocator = std.mem.Allocator;

const ziggurat = @import("ziggurat");
const duct = @import("duct");

pub fn flatLenIterate(
    accumulator: usize,
    element: usize,
    _: usize,
    _: []const usize,
) usize {
    return accumulator * element;
}

pub fn flatLen(shape: anytype) usize {
    if (shape.len == 0) return 1;
    return duct.all.get.reduce(shape, flatLenIterate);
}

pub fn flatIndex(
    strides: anytype,
    indices: anytype,
) ziggurat.sign(.seq(&.{
    .any(&.{
        .is_array(.{ .child = .is_int(.{ .signedness = .{ .unsigned = true } }) }),
        .is_vector(.{ .child = .is_int(.{ .signedness = .{ .unsigned = true } }) }),
        .is_pointer(.{
            .child = .is_int(.{ .signedness = .{ .unsigned = true } }),
            .size = .{ .slice = true },
        }),
    }),
    .any(&.{
        .is_array(.{ .child = .is_int(.{ .signedness = .{ .unsigned = true } }) }),
        .is_vector(.{ .child = .is_int(.{ .signedness = .{ .unsigned = true } }) }),
        .is_pointer(.{
            .child = .is_int(.{ .signedness = .{ .unsigned = true } }),
            .size = .{ .slice = true },
        }),
    }),
}))(.{
    @TypeOf(strides),
    @TypeOf(indices),
})(usize) {
    if (indices.len == 0) return 0;

    var result: usize = 0;

    for (1..strides.len + 1) |i| {
        const index = strides.len - i;
        result += duct.get.at(indices, index) * duct.get.at(strides, index);
    }

    return result;
}

pub fn dimIndex(allocator: Allocator, strides: []const usize, index: usize) ![]const usize {
    var result = try duct.new.zeroes(allocator, usize, strides.len);
    var value = index;
    var dim: usize = 0;

    while (dim < strides.len) : (dim += 1) {
        result[dim] = try std.math.divFloor(usize, value, strides[dim]);
        value -= result[dim] * strides[dim];
    }

    return result;
}

pub const ReshapeError = error{MismatchedLengths};

pub fn initReshape(
    allocator: Allocator,
    shape: anytype,
    new_shape: anytype,
) ziggurat.sign(.seq(&.{
    .any(&.{
        .is_array(.{ .child = .is_int(.{ .signedness = .{ .unsigned = true } }) }),
        .is_vector(.{ .child = .is_int(.{ .signedness = .{ .unsigned = true } }) }),
        .is_pointer(.{
            .child = .is_int(.{ .signedness = .{ .unsigned = true } }),
            .size = .{ .slice = true },
        }),
    }),
    .any(&.{
        .is_array(.{ .child = .is_int(.{ .signedness = .{ .unsigned = true } }) }),
        .is_vector(.{ .child = .is_int(.{ .signedness = .{ .unsigned = true } }) }),
        .is_pointer(.{
            .child = .is_int(.{ .signedness = .{ .unsigned = true } }),
            .size = .{ .slice = true },
        }),
    }),
}))(.{
    @TypeOf(shape),
    @TypeOf(new_shape),
})((Allocator.Error || ReshapeError)![]const usize) {
    const new_shape_rank = duct.get.len(new_shape);

    if (flatLen(shape) != flatLen(new_shape))
        return ReshapeError.MismatchedLengths;

    const result = try allocator.alloc(usize, new_shape_rank);

    for (0..new_shape_rank) |index| {
        result[index] = duct.get.at(new_shape, index);
    }

    return result;
}

pub fn initStrides(
    allocator: Allocator,
    shape: anytype,
) ziggurat.sign(.any(&.{
    .is_array(.{ .child = .is_int(.{ .signedness = .{ .unsigned = true } }) }),
    .is_vector(.{ .child = .is_int(.{ .signedness = .{ .unsigned = true } }) }),
    .is_pointer(.{
        .child = .is_int(.{ .signedness = .{ .unsigned = true } }),
        .size = .{ .slice = true },
    }),
}))(
    @TypeOf(shape),
)(Allocator.Error![]usize) {
    const len = duct.get.len(shape);

    if (len == 0) {
        const result = try allocator.alloc(usize, 1);
        result[0] = 0;
        return result;
    }

    const result = try allocator.alloc(usize, len);
    var product: usize = 1;

    for (1..len + 1) |i| {
        const index = len - i;
        result[index] = product;
        product *= duct.get.at(shape, index);
    }

    return result;
}
