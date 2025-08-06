pub const dtype = enum {
    int8,
    int16,
    int32,
    int64,
    int128,
    uint8,
    uint16,
    uint32,
    uint64,
    uint128,
    float32,
    float64,
    float128,

    pub fn toType(self: dtype) type {
        return switch (self) {
            .int8 => i8,
            .int16 => i16,
            .int32 => i32,
            .int64 => i64,
            .int128 => i128,
            .uint8 => u8,
            .uint16 => u16,
            .uint32 => u32,
            .uint64 => u64,
            .uint128 => u128,
            .float16 => f16,
            .float32 => f32,
            .float64 => f64,
            .float128 => f128,
        };
    }

    pub fn fromType(T: type) !dtype {
        return switch (T) {
            i8 => .int8,
            i16 => .int16,
            i32 => .int32,
            i64 => .int64,
            i128 => .int128,
            u8 => .uint8,
            u16 => .uint16,
            u32 => .uint32,
            u64 => .uint64,
            u128 => .uint128,
            f16 => .float16,
            f32 => .float32,
            f64 => .float64,
            f128 => .float128,
            else => return error.UnsupportedType,
        };
    }
};
