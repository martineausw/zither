pub const all = @import("ops/all.zig");
pub const reduce = @import("ops/reduce.zig");
pub const contract = @import("ops/contract.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
