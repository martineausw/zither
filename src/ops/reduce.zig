pub const sum = @import("reduce/sum.zig").sum;
pub const prod = @import("reduce/prod.zig").prod;

test {
    @import("std").testing.refAllDecls(@This());
}
