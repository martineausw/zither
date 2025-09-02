const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

const duct = @import("duct");
const Tensor = @import("tensor.zig").Tensor;

pub fn add(allocator: Allocator, comptime T: type, tensor_0: *const Tensor(T), tensor_1: *const Tensor(T)) !Tensor(T) {
    if (tensor_0.shape != tensor_1.shape) return error.MismatchedShape;
    if (tensor_0.strides != tensor_1.strides) return error.MismatchedStrides;

    return try duct.iterate.new.add(allocator, T, tensor_0.buffer, tensor_1.buffer);
}

pub fn sub(allocator: Allocator, comptime T: type, tensor_0: *const Tensor(T), tensor_1: *const Tensor(T)) !Tensor(T) {
    if (tensor_0.shape != tensor_1.shape) return error.MismatchedShape;
    if (tensor_0.strides != tensor_1.strides) return error.MismatchedStrides;

    return try duct.iterate.new.sub(allocator, T, tensor_0.buffer, tensor_1.buffer);
}

pub fn mul(allocator: Allocator, comptime T: type, tensor_0: *const Tensor(T), tensor_1: *const Tensor(T)) !Tensor(T) {
    if (tensor_0.shape != tensor_1.shape) return error.MismatchedShape;
    if (tensor_0.strides != tensor_1.strides) return error.MismatchedStrides;

    return try duct.iterate.new.mul(allocator, T, tensor_0.buffer, tensor_1.buffer);
}

pub fn div(allocator: Allocator, comptime T: type, tensor_0: *const Tensor(T), tensor_1: *const Tensor(T)) !Tensor(T) {
    if (tensor_0.shape != tensor_1.shape) return error.MismatchedShape;
    if (tensor_0.strides != tensor_1.strides) return error.MismatchedStrides;

    return try duct.iterate.new.div(allocator, T, tensor_0.buffer, tensor_1.buffer);
}

pub fn divFloor(allocator: Allocator, comptime T: type, tensor_0: *const Tensor(T), tensor_1: *const Tensor(T)) !Tensor(T) {
    if (tensor_0.shape != tensor_1.shape) return error.MismatchedShape;
    if (tensor_0.strides != tensor_1.strides) return error.MismatchedStrides;

    return try duct.iterate.new.divFloor(allocator, T, tensor_0.buffer, tensor_1.buffer);
}

pub fn divCeil(allocator: Allocator, comptime T: type, tensor_0: *const Tensor(T), tensor_1: *const Tensor(T)) !Tensor(T) {
    if (tensor_0.shape != tensor_1.shape) return error.MismatchedShape;
    if (tensor_0.strides != tensor_1.strides) return error.MismatchedStrides;

    return try duct.iterate.new.divCeil(allocator, T, tensor_0.buffer, tensor_1.buffer);
}
