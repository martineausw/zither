const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

const ziggurat = @import("ziggurat");
const duct = @import("duct");
const utils = @import("utils.zig");

const tensor_element: ziggurat.Prototype = .any(&.{
    .is_int(.{}),
    .is_float(.{}),
    .is_bool,
});

const Tensor = @import("tensor.zig").Tensor;

pub fn TensorView(comptime T: type) ziggurat.sign(tensor_element)(T)(type) {
    return struct {
        buffer: []const T,
        shape: []usize,
        strides: []usize,
        allocator: Allocator,

        pub fn from(tensor: *const Tensor(T)) Allocator.Error!TensorView(T) {
            return .{
                .buffer = tensor.*.buffer,
                .shape = try duct.new.copy(tensor.*.allocator, tensor.*.shape),
                .strides = try utils.initStrides(tensor.*.allocator, tensor.*.shape),
                .allocator = tensor.*.allocator,
            };
        }

        pub fn deinit(self: *const TensorView(T)) void {
            self.allocator.free(self.shape);
            self.allocator.free(self.strides);
        }

        pub fn rank(self: *const TensorView(T)) usize {
            return self.shape.len;
        }

        pub fn at(self: *const TensorView(T), indices: []const usize) T {
            return self.buffer[utils.flatIndex(self.strides, indices)];
        }

        pub fn item(self: *const TensorView(T)) ?T {
            if (self.shape.len == 1 and self.shape[0] == 1) return self.buffer[0];
            return null;
        }

        pub fn flatten(self: *TensorView(T)) Allocator.Error!void {
            const shape = utils.initReshape(
                self.allocator,
                self.shape,
                @as([]const usize, &[1]usize{duct.iterate.get.reduce(self.shape, utils.flatLen)}),
            ) catch |err| return switch (err) {
                Allocator.Error.OutOfMemory => Allocator.Error.OutOfMemory,
                utils.ReshapeError.MismatchedLengths => unreachable,
            };
            const strides = try utils.initStrides(self.allocator, shape);
            self.allocator.free(self.shape);
            self.allocator.free(self.strides);
            self.shape = @constCast(shape);
            self.strides = @constCast(strides);
        }

        pub fn broadcastTo(self: *TensorView(T), target_shape: []const usize) void {
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

        pub fn transpose(self: *TensorView(T), axes: []const usize) !void {
            try duct.set.transpose(self.allocator, self.shape, axes);
            try duct.set.transpose(self.allocator, self.strides, axes);
        }

        pub fn reshape(self: *TensorView(T), new_shape: []const usize) (Allocator.Error || utils.ReshapeError)!void {
            const shape = try utils.initReshape(self.allocator, self.shape, new_shape);
            const strides = try utils.initStrides(self.allocator, shape);
            self.allocator.free(self.shape);
            self.allocator.free(self.strides);
            self.shape = @constCast(shape);
            self.strides = @constCast(strides);
        }

        pub fn squeeze(self: *TensorView(T), axis: usize) Allocator.Error!void {
            if (self.shape[axis] != 1) return;
            _ = try duct.set.remove(self.allocator, &self.shape, axis);
            const strides = try utils.initStrides(self.allocator, self.shape);
            self.allocator.free(self.strides);
            self.strides = @constCast(strides);
        }

        pub fn unsqueeze(self: *TensorView(T), axis: usize) Allocator.Error!void {
            try duct.set.insert(self.allocator, &self.shape, axis, 1);
            const strides = try utils.initStrides(self.allocator, self.shape);
            self.allocator.free(self.strides);
            self.strides = @constCast(strides);
        }
    };
}

test {
    const tensor = try Tensor(usize).arange(testing.allocator, 0, 6, 1);
    defer tensor.deinit();
    var tensor_view = try TensorView(usize).from(&tensor);
    defer tensor_view.deinit();

    try testing.expectEqualSlices(usize, &.{1}, tensor_view.strides);
    try testing.expectEqualSlices(usize, &.{6}, tensor_view.shape);

    try testing.expectEqual(0, tensor_view.at(&.{0}));
    try testing.expectEqual(1, tensor_view.at(&.{1}));
    try testing.expectEqual(2, tensor_view.at(&.{2}));
    try testing.expectEqual(3, tensor_view.at(&.{3}));
    try testing.expectEqual(4, tensor_view.at(&.{4}));
    try testing.expectEqual(5, tensor_view.at(&.{5}));

    try tensor_view.reshape(&.{ 3, 2, 1 });

    try testing.expectEqualSlices(usize, &.{ 3, 2, 1 }, tensor_view.shape);
    try testing.expectEqualSlices(usize, &.{ 2, 1, 1 }, tensor_view.strides);

    try testing.expectEqual(0, tensor_view.at(&.{ 0, 0, 0 }));
    try testing.expectEqual(1, tensor_view.at(&.{ 0, 1, 0 }));
    try testing.expectEqual(2, tensor_view.at(&.{ 1, 0, 0 }));
    try testing.expectEqual(3, tensor_view.at(&.{ 1, 1, 0 }));
    try testing.expectEqual(4, tensor_view.at(&.{ 2, 0, 0 }));
    try testing.expectEqual(5, tensor_view.at(&.{ 2, 1, 0 }));

    try tensor_view.squeeze(2);

    try testing.expectEqualSlices(usize, &.{ 3, 2 }, tensor_view.shape);
    try testing.expectEqualSlices(usize, &.{ 2, 1 }, tensor_view.strides);

    try testing.expectEqual(0, tensor_view.at(&.{ 0, 0 }));
    try testing.expectEqual(1, tensor_view.at(&.{ 0, 1 }));
    try testing.expectEqual(2, tensor_view.at(&.{ 1, 0 }));
    try testing.expectEqual(3, tensor_view.at(&.{ 1, 1 }));
    try testing.expectEqual(4, tensor_view.at(&.{ 2, 0 }));
    try testing.expectEqual(5, tensor_view.at(&.{ 2, 1 }));

    try tensor_view.unsqueeze(0);

    try testing.expectEqualSlices(usize, &.{ 1, 3, 2 }, tensor_view.shape);
    try testing.expectEqualSlices(usize, &.{ 6, 2, 1 }, tensor_view.strides);

    try testing.expectEqual(0, tensor_view.at(&.{ 0, 0, 0 }));
    try testing.expectEqual(1, tensor_view.at(&.{ 0, 0, 1 }));
    try testing.expectEqual(2, tensor_view.at(&.{ 0, 1, 0 }));
    try testing.expectEqual(3, tensor_view.at(&.{ 0, 1, 1 }));
    try testing.expectEqual(4, tensor_view.at(&.{ 0, 2, 0 }));
    try testing.expectEqual(5, tensor_view.at(&.{ 0, 2, 1 }));

    try tensor_view.transpose(&.{ 0, 2, 1 });

    try testing.expectEqualSlices(usize, &.{ 1, 2, 3 }, tensor_view.shape);
    try testing.expectEqualSlices(usize, &.{ 6, 1, 2 }, tensor_view.strides);

    try testing.expectEqual(0, tensor_view.at(&.{ 0, 0, 0 }));
    try testing.expectEqual(1, tensor_view.at(&.{ 0, 1, 0 }));
    try testing.expectEqual(2, tensor_view.at(&.{ 0, 0, 1 }));
    try testing.expectEqual(3, tensor_view.at(&.{ 0, 1, 1 }));
    try testing.expectEqual(4, tensor_view.at(&.{ 0, 0, 2 }));
    try testing.expectEqual(5, tensor_view.at(&.{ 0, 1, 2 }));

    try tensor_view.flatten();

    try testing.expectEqualSlices(usize, &.{6}, tensor_view.shape);
    try testing.expectEqualSlices(usize, &.{1}, tensor_view.strides);

    try testing.expectEqual(0, tensor_view.at(&.{0}));
    try testing.expectEqual(1, tensor_view.at(&.{1}));
    try testing.expectEqual(2, tensor_view.at(&.{2}));
    try testing.expectEqual(3, tensor_view.at(&.{3}));
    try testing.expectEqual(4, tensor_view.at(&.{4}));
    try testing.expectEqual(5, tensor_view.at(&.{5}));
}
