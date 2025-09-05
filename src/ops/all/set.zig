const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

const duct = @import("duct");
const Tensor = @import("../../tensor.zig").Tensor;
const utils = @import("../../utils.zig");

pub fn map(
    comptime T: type,
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
        T,
        &dest.*.buffer,
        aux.*.buffer,
        func,
    );
}

pub fn add(
    comptime T: type,
    dest: *Tensor(T),
    aux: *const Tensor(T),
) void {
    map(
        T,
        dest,
        aux,
        duct.all.ops.elm.Element(T, Tensor(T), Tensor(T)).add,
    );
}

pub fn sub(
    comptime T: type,
    dest: *Tensor(T),
    aux: *const Tensor(T),
) void {
    map(
        T,
        dest,
        aux,
        duct.all.ops.elm.Element(T, Tensor(T), Tensor(T)).sub,
    );
}

pub fn mul(
    comptime T: type,
    dest: *const Tensor(T),
    aux: *const Tensor(T),
) void {
    map(
        T,
        dest,
        aux,
        duct.all.ops.elm.Element(T, Tensor(T), Tensor(T)).mul,
    );
}

pub fn div(
    comptime T: type,
    dest: *const Tensor(T),
    aux: *const Tensor(T),
) void {
    map(
        T,
        dest,
        aux,
        duct.all.ops.elm.Element(T, Tensor(T), Tensor(T)).div,
    );
}

pub fn divFloor(
    comptime T: type,
    dest: *const Tensor(T),
    aux: *const Tensor(T),
) void {
    map(
        T,
        dest,
        aux,
        duct.all.ops.elm.Element(T, Tensor(T), Tensor(T)).divFloor,
    );
}

pub fn divCeil(
    comptime T: type,
    dest: *const Tensor(T),
    aux: *const Tensor(T),
) void {
    map(
        T,
        dest,
        aux,
        duct.all.ops.elm.Element(T, Tensor(T), Tensor(T)).divCeil,
    );
}
