from reproduce.codec_content_rate import screen_gate

def test_large_non_vcl_overhead_cannot_hide_content_rate_error():
    cfg={'rate_median_error_max':.35,'rate_p90_error_max':.8,'rate_direction_min':.75}
    # Total looks close because 99% of rate is fixed; content estimate is 10x.
    rows=[dict(real_bpp=100.,relaxed_bpp=109.,real_content_bpp=1.,relaxed_content_bpp=10.)]*8
    assert not screen_gate(rows,dict(aligned=8,untied=8),cfg)['passed']
