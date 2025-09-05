const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

const duct = @import("duct");
const Tensor = @import("../../tensor.zig").Tensor;
const utils = @import("../../utils.zig");

pub fn map(
    allocator: Allocator,
    comptime T: type,
    tensor_0: *const Tensor(T),
    tensor_1: *const Tensor(T),
    func: *const fn (
        elements: struct { T, T },
        index: usize,
        data: struct { []const T, []const T },
    ) T,
) !Tensor(T) {
    if (tensor_0.shape != tensor_1.shape) return error.MismatchedShape;

    const buffer = duct.all.ops.elm.new.map(
        allocator,
        T,
        tensor_0.buffer,
        tensor_1.buffer,
        func,
    );

    return .{
        .buffer = buffer,
        .shape = tensor_0.shape,
        .strides = try utils.initStrides(allocator, tensor_0.shape),
        .allocator = allocator,
    };
}

pub fn add(
    allocator: Allocator,
    comptime T: type,
    tensor_0: *const Tensor(T),
    tensor_1: *const Tensor(T),
) !Tensor(T) {
    return map(
        allocator,
        T,
        tensor_0,
        tensor_1,
        duct.all.ops.elm.Element(T, Tensor(T), Tensor(T)).add,
    );
}

pub fn sub(
    allocator: Allocator,
    comptime T: type,
    tensor_0: *const Tensor(T),
    tensor_1: *const Tensor(T),
) !Tensor(T) {
    return map(
        allocator,
        T,
        tensor_0,
        tensor_1,
        duct.all.ops.elm.Element(T, Tensor(T), Tensor(T)).sub,
    );
}

pub fn mul(
    allocator: Allocator,
    comptime T: type,
    tensor_0: *const Tensor(T),
    tensor_1: *const Tensor(T),
) !Tensor(T) {
    return map(
        allocator,
        T,
        tensor_0,
        tensor_1,
        duct.all.ops.elm.Element(T, Tensor(T), Tensor(T)).mul,
    );
}

pub fn div(
    allocator: Allocator,
    comptime T: type,
    tensor_0: *const Tensor(T),
    tensor_1: *const Tensor(T),
) !Tensor(T) {
    return map(
        allocator,
        T,
        tensor_0,
        tensor_1,
        duct.all.ops.elm.Element(T, Tensor(T), Tensor(T)).div,
    );
}

pub fn divFloor(
    allocator: Allocator,
    comptime T: type,
    tensor_0: *const Tensor(T),
    tensor_1: *const Tensor(T),
) !Tensor(T) {
    return map(
        allocator,
        T,
        tensor_0,
        tensor_1,
        duct.all.ops.elm.Element(T, Tensor(T), Tensor(T)).divFloor,
    );
}

pub fn divCeil(
    allocator: Allocator,
    comptime T: type,
    tensor_0: *const Tensor(T),
    tensor_1: *const Tensor(T),
) !Tensor(T) {
    return map(
        allocator,
        T,
        tensor_0,
        tensor_1,
        duct.all.ops.elm.Element(T, Tensor(T), Tensor(T)).divCeil,
    );
}
