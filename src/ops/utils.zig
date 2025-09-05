const std = @import("std");
const Allocator = std.mem.Allocator;

const duct = @import("duct");

const Tensor = @import("../tensor.zig").Tensor;

pub fn incrementIndices(strides: []const usize, indices: *[]usize, iteration: usize) !void {
    var value = iteration;

    for (0..indices.len) |index| {
        indices.*[index] = @intCast(@divFloor(value, strides[index]));
        value -= strides[index] * @divFloor(value, strides[index]);
    }
}

pub fn contract(comptime T: type) type {
    return struct {
        pub fn createShape(
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

        pub fn mapAccessorIndices(
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

        pub fn calculateElement(
            func: *const fn (
                accumulator: T,
                elements: struct { T, T },
                indices: struct { []const usize, []const usize },
                tensors: struct { *const Tensor(T), *const Tensor(T) },
            ) T,
            initial_value: T,
            tensor_0: Tensor(T),
            axes_0: []const usize,
            indices_0: *[]usize,
            tensor_1: Tensor(T),
            axes_1: []const usize,
            indices_1: *[]usize,
            depth: usize,
        ) T {
            var result: T = initial_value;
            if (depth == axes_0.len - 1) {
                for (0..tensor_0.shape[axes_0[depth]]) |dim| {
                    indices_0.*[axes_0[depth]] = dim;
                    indices_1.*[axes_1[depth]] = dim;
                    result = func(
                        result,
                        .{ tensor_0.at(indices_0.*), tensor_1.at(indices_1.*) },
                        .{ indices_0.*, indices_1.* },
                        .{ &tensor_0, &tensor_1 },
                    );
                }
            } else {
                for (0..tensor_0.shape[axes_0[depth]]) |dim| {
                    indices_0.*[axes_0[depth]] = dim;
                    indices_1.*[axes_1[depth]] = dim;
                    result = func(
                        result,
                        .{
                            calculateElement(
                                func,
                                initial_value,
                                tensor_0,
                                axes_0,
                                indices_0,
                                tensor_1,
                                axes_1,
                                indices_1,
                                depth + 1,
                            ),
                            1,
                        },
                        .{ indices_0.*, indices_1.* },
                        .{ &tensor_0, &tensor_1 },
                    );
                }
            }

            return result;
        }

        pub fn tensordot(
            accumulator: T,
            elements: struct { T, T },
            _: struct { []const usize, []const usize },
            _: struct { *const Tensor(T), *const Tensor(T) },
        ) T {
            return accumulator + elements.@"0" * elements.@"1";
        }
    };
}

pub fn reduce(comptime T: type) type {
    return struct {
        pub fn createShape(
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

        pub fn mapAccessorIndices(
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

        pub fn calculateElement(
            initial: T,
            tensor: Tensor(T),
            axes: []const usize,
            indices: *[]usize,
            func: *const fn (
                accumulator: T,
                element: T,
                indices: []const usize,
                tensor: *const Tensor(T),
            ) T,
            depth: usize,
        ) T {
            var result: T = initial;
            if (depth == axes.len - 1) {
                for (0..tensor.shape[axes[depth]]) |dim| {
                    indices.*[axes[depth]] = dim;
                    result = func(
                        result,
                        tensor.at(indices.*),
                        indices.*,
                        &tensor,
                    );
                }
            } else {
                for (0..tensor.shape[axes[depth]]) |dim| {
                    indices.*[axes[depth]] = dim;

                    result = func(
                        result,
                        calculateElement(
                            initial,
                            tensor,
                            axes,
                            indices,
                            func,
                            depth + 1,
                        ),
                        indices.*,
                        &tensor,
                    );
                }
            }

            return result;
        }

        pub fn sum(
            accumulator: T,
            element: T,
            _: []const usize,
            _: *const Tensor(T),
        ) T {
            return accumulator + element;
        }

        pub fn product(
            accumulator: T,
            element: T,
            _: []const usize,
            _: *const Tensor(T),
        ) T {
            return accumulator * element;
        }
    };
}
