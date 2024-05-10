@fragment
fn fragment(
    @builtin(primitive_index) i: u32,
) -> @location(0) vec4<f32> {
    let hue = fract(f32(i) * 1.71) * 6.;
    return vec4(
        -1. + abs(hue - 3.),
         2. - abs(hue - 2.),
         2. - abs(hue - 4.),
         1.
    );
}
