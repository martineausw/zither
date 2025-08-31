const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

const ziggurat = @import("ziggurat");
const duct = @import("duct");
const utils = @import("utils.zig");

const TensorView = @import("tensor_view.zig").TensorView;

const tensor_element: ziggurat.Prototype = .any(&.{
    .is_int(.{}),
    .is_float(.{}),
    .is_bool,
});

pub fn Tensor(comptime T: type) ziggurat.sign(tensor_element)(T)(type) {
    return struct {
        buffer: []T,
        shape: []usize,
        strides: []usize,
        allocator: Allocator,

        pub fn init(
            allocator: Allocator,
            shape: []const usize,
        ) Allocator.Error!Tensor(T) {
            return .{
                .buffer = try duct.new.zeroes(
                    allocator,
                    T,
                    duct.iterate.get.reduce(shape, utils.flatLen),
                ),
                .shape = try duct.new.copy(allocator, shape),
                .strides = try utils.initStrides(allocator, shape),
                .allocator = allocator,
            };
        }

        pub fn identity(
            allocator: Allocator,
            n: usize,
        ) Allocator.Error!Tensor(T) {
            const buffer = try allocator.alloc(usize, n * n);
            for (0..buffer.len) |index| {
                buffer[index] = 0;
            }

            const shape = try allocator.alloc(usize, 2);
            shape[0] = n;
            shape[1] = n;

            const strides = try utils.initStrides(allocator, shape);

            for (0..n) |index| {
                buffer[
                    utils.flatIndex(strides, [_]usize{ index, index })
                ] = 1;
            }

            return .{
                .buffer = buffer,
                .shape = shape,
                .strides = strides,
                .allocator = allocator,
            };
        }

        pub fn zeroes(
            allocator: Allocator,
            shape: []const usize,
        ) Allocator.Error!Tensor(T) {
            return full(allocator, shape, 0);
        }

        pub fn ones(
            allocator: Allocator,
            shape: []const usize,
        ) Allocator.Error!Tensor(T) {
            return full(allocator, shape, 1);
        }

        pub fn full(
            allocator: Allocator,
            shape: []const usize,
            value: T,
        ) Allocator.Error!Tensor(T) {
            return .{
                .buffer = try duct.new.fill(
                    allocator,
                    T,
                    duct.iterate.get.reduce(shape, utils.flatLen),
                    value,
                ),
                .shape = try duct.new.copy(allocator, shape),
                .strides = try utils.initStrides(allocator, shape),
                .allocator = allocator,
            };
        }

        pub fn random(
            allocator: Allocator,
            shape: []const usize,
            min: T,
            max: T,
        ) Allocator.Error!Tensor(T) {
            const rand = std.crypto.random;

            const rand_type = switch (@typeInfo(T)) {
                .int => |info| std.meta.Float(info.bits),
                .float => T,
                .bool => T,
                else => unreachable,
            };

            const range = max - min;

            const buffer = try allocator.alloc(T, duct.iterate.get.reduce(shape, utils.flatLen));

            for (0..buffer.len) |index| {
                const value = rand.float(rand_type);

                buffer[index] = switch (@typeInfo(T)) {
                    .int => @intFromFloat(value * range + min),
                    .float => value * range + min,
                    .bool => value > 0.5,
                    else => unreachable,
                };
            }

            return .{
                .buffer = buffer,
                .shape = try duct.new.copy(allocator, shape),
                .strides = try utils.initStrides(allocator, shape),
                .allocator = allocator,
            };
        }

        pub fn arange(
            allocator: Allocator,
            start: T,
            end: T,
            step: T,
        ) anyerror!Tensor(T) {
            const len = try std.math.divFloor(
                usize,
                end - start,
                step,
            );
            const shape = try duct.new.fill(allocator, T, 1, len);
            return .{
                .buffer = try duct.new.arange(allocator, T, len, start, step),
                .shape = shape,
                .strides = try utils.initStrides(allocator, shape),
                .allocator = allocator,
            };
        }

        pub fn view(self: *const Tensor(T)) Allocator.Error!TensorView(T) {
            return TensorView(T).from(self);
        }

        pub fn deinit(self: *const Tensor(T)) void {
            self.allocator.free(self.buffer);
            self.allocator.free(self.shape);
            self.allocator.free(self.strides);
        }

        pub fn rank(self: *const Tensor(T)) usize {
            return self.shape.len;
        }

        pub fn at(
            self: *const Tensor(T),
            indices: []const usize,
        ) T {
            return self.buffer[utils.flatIndex(self.strides, indices)];
        }

        pub fn set(
            self: *Tensor(T),
            indices: []const usize,
            value: T,
        ) void {
            self.buffer[utils.flatIndex(self.strides, indices)] = value;
        }

        pub fn item(self: *const Tensor(T)) ?T {
            if (self.shape.len == 1 and self.shape[0] == 1) return self.buffer[0];
            return null;
        }

        pub fn flatten(self: *Tensor(T)) Allocator.Error!void {
            const shape = utils.initReshape(
                self.allocator,
                self.shape,
                @as([]const usize, &[1]usize{duct.iterate.get.reduce(self.shape, utils.flatLen)}),
            ) catch |err| return switch (err) {
                utils.ReshapeError.MismatchedLengths => unreachable,
                Allocator.Error.OutOfMemory => return Allocator.Error.OutOfMemory,
            };
            const strides = try utils.initStrides(self.allocator, shape);
            self.allocator.free(self.shape);
            self.allocator.free(self.strides);
            self.shape = @constCast(shape);
            self.strides = @constCast(strides);
        }

        pub fn broadcastTo(
            self: *Tensor(T),
            target_shape: []const usize,
        ) void {
            for (self.shape, 0..) |value, index| {
                if (target_shape[index] == value) continue;
                if (target_shape[index] != value and value != 1) {
                    return;
                }
            }

            for (self.shape, 0..) |value, index| {
                if (target_shape[index] == value) continue;
                if (target_shape[index] != value and value == 1) {
                    self.shape[index] = target_shape[index];
                    self.strides[index] = 0;
                }
            }
        }

        pub fn transpose(
            self: *Tensor(T),
            axes: []const usize,
        ) Allocator.Error!void {
            try duct.set.transpose(self.allocator, self.shape, axes);
            try duct.set.transpose(self.allocator, self.strides, axes);
        }

        pub fn reshape(
            self: *Tensor(T),
            new_shape: []const usize,
        ) (Allocator.Error || utils.ReshapeError)!void {
            const shape = try utils.initReshape(self.allocator, self.shape, new_shape);
            const strides = try utils.initStrides(self.allocator, shape);
            self.allocator.free(self.shape);
            self.allocator.free(self.strides);
            self.shape = @constCast(shape);
            self.strides = @constCast(strides);
        }

        pub fn squeeze(
            self: *Tensor(T),
            axis: usize,
        ) Allocator.Error!void {
            if (self.shape[axis] != 1) return;
            _ = try duct.set.remove(self.allocator, &self.shape, axis);
            const strides = utils.initStrides(self.allocator, self.shape) catch unreachable;
            self.allocator.free(self.strides);
            self.strides = @constCast(strides);
        }

        pub fn unsqueeze(
            self: *Tensor(T),
            axis: usize,
        ) Allocator.Error!void {
            try duct.set.insert(self.allocator, &self.shape, axis, 1);
            const strides = try utils.initStrides(self.allocator, self.shape);
            self.allocator.free(self.strides);
            self.strides = @constCast(strides);
        }
    };
}

test {
    var tensor_0 = try Tensor(usize).init(testing.allocator, &.{6});
    defer tensor_0.deinit();

    const tensor_view = try tensor_0.view();
    tensor_view.deinit();

    for (0..tensor_0.buffer.len) |index| {
        tensor_0.set(&.{index}, index);
    }

    var tensor_1 = try Tensor(usize).zeroes(testing.allocator, &.{100});
    defer tensor_1.deinit();

    for (tensor_1.buffer) |value| {
        try testing.expectEqual(0, value);
    }

    var tensor_2 = try Tensor(usize).ones(testing.allocator, &.{100});
    defer tensor_2.deinit();

    for (tensor_2.buffer) |value| {
        try testing.expectEqual(1, value);
    }

    var tensor_3 = try Tensor(usize).full(testing.allocator, &.{100}, 2);
    defer tensor_3.deinit();

    for (tensor_3.buffer) |value| {
        try testing.expectEqual(2, value);
    }

    var tensor_4 = try Tensor(usize).arange(testing.allocator, 0, 200, 2);
    defer tensor_4.deinit();

    for (tensor_4.buffer, 0..) |value, index| {
        try testing.expectEqual(index * 2, value);
    }

    var tensor_5 = try Tensor(usize).identity(testing.allocator, 2);
    defer tensor_5.deinit();

    for (0..2) |value| {
        try testing.expectEqual(1, tensor_5.at(&.{ value, value }));
    }

    try testing.expectEqualSlices(usize, &.{1}, tensor_0.strides);
    try testing.expectEqualSlices(usize, &.{6}, tensor_0.shape);

    try testing.expectEqual(0, tensor_0.at(&.{0}));
    try testing.expectEqual(1, tensor_0.at(&.{1}));
    try testing.expectEqual(2, tensor_0.at(&.{2}));
    try testing.expectEqual(3, tensor_0.at(&.{3}));
    try testing.expectEqual(4, tensor_0.at(&.{4}));
    try testing.expectEqual(5, tensor_0.at(&.{5}));

    try tensor_0.reshape(&.{ 3, 2, 1 });

    try testing.expectEqualSlices(usize, &.{ 3, 2, 1 }, tensor_0.shape);
    try testing.expectEqualSlices(usize, &.{ 2, 1, 1 }, tensor_0.strides);

    try testing.expectEqual(0, tensor_0.at(&.{ 0, 0, 0 }));
    try testing.expectEqual(1, tensor_0.at(&.{ 0, 1, 0 }));
    try testing.expectEqual(2, tensor_0.at(&.{ 1, 0, 0 }));
    try testing.expectEqual(3, tensor_0.at(&.{ 1, 1, 0 }));
    try testing.expectEqual(4, tensor_0.at(&.{ 2, 0, 0 }));
    try testing.expectEqual(5, tensor_0.at(&.{ 2, 1, 0 }));

    try tensor_0.squeeze(2);

    try testing.expectEqualSlices(usize, &.{ 3, 2 }, tensor_0.shape);
    try testing.expectEqualSlices(usize, &.{ 2, 1 }, tensor_0.strides);

    try testing.expectEqual(0, tensor_0.at(&.{ 0, 0 }));
    try testing.expectEqual(1, tensor_0.at(&.{ 0, 1 }));
    try testing.expectEqual(2, tensor_0.at(&.{ 1, 0 }));
    try testing.expectEqual(3, tensor_0.at(&.{ 1, 1 }));
    try testing.expectEqual(4, tensor_0.at(&.{ 2, 0 }));
    try testing.expectEqual(5, tensor_0.at(&.{ 2, 1 }));

    try tensor_0.unsqueeze(0);

    try testing.expectEqualSlices(usize, &.{ 1, 3, 2 }, tensor_0.shape);
    try testing.expectEqualSlices(usize, &.{ 6, 2, 1 }, tensor_0.strides);

    try testing.expectEqual(0, tensor_0.at(&.{ 0, 0, 0 }));
    try testing.expectEqual(1, tensor_0.at(&.{ 0, 0, 1 }));
    try testing.expectEqual(2, tensor_0.at(&.{ 0, 1, 0 }));
    try testing.expectEqual(3, tensor_0.at(&.{ 0, 1, 1 }));
    try testing.expectEqual(4, tensor_0.at(&.{ 0, 2, 0 }));
    try testing.expectEqual(5, tensor_0.at(&.{ 0, 2, 1 }));

    try tensor_0.transpose(&.{ 0, 2, 1 });

    try testing.expectEqualSlices(usize, &.{ 1, 2, 3 }, tensor_0.shape);
    try testing.expectEqualSlices(usize, &.{ 6, 1, 2 }, tensor_0.strides);

    try testing.expectEqual(0, tensor_0.at(&.{ 0, 0, 0 }));
    try testing.expectEqual(1, tensor_0.at(&.{ 0, 1, 0 }));
    try testing.expectEqual(2, tensor_0.at(&.{ 0, 0, 1 }));
    try testing.expectEqual(3, tensor_0.at(&.{ 0, 1, 1 }));
    try testing.expectEqual(4, tensor_0.at(&.{ 0, 0, 2 }));
    try testing.expectEqual(5, tensor_0.at(&.{ 0, 1, 2 }));

    try tensor_0.flatten();

    try testing.expectEqualSlices(usize, &.{6}, tensor_0.shape);
    try testing.expectEqualSlices(usize, &.{1}, tensor_0.strides);

    try testing.expectEqual(0, tensor_0.at(&.{0}));
    try testing.expectEqual(1, tensor_0.at(&.{1}));
    try testing.expectEqual(2, tensor_0.at(&.{2}));
    try testing.expectEqual(3, tensor_0.at(&.{3}));
    try testing.expectEqual(4, tensor_0.at(&.{4}));
    try testing.expectEqual(5, tensor_0.at(&.{5}));
}
